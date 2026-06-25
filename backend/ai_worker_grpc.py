from __future__ import annotations

import logging
import os
import queue
import socket
import time
import sys
import threading
import uuid
import warnings

# Suppress numpy FutureWarning from insightface
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

import cv2
import grpc
import numpy as np
import concurrent.futures

# Add current directory to path to ensure protobuf imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import facerec_pb2
import facerec_pb2_grpc
from collections import defaultdict, Counter
from app.face_engine import face_engine, compute_sharpness, SHARPNESS_THRESHOLD
from app.face_restorer import face_restorer
from app.license_plate import LicensePlateEngine, PlateResult, CharDet
from app.license_plate.validate import LicensePlateValidator

license_plate_engine: LicensePlateEngine | None = None

# Persistent identity for this worker process. Set WORKER_NAME env var to a
# stable name (e.g. "gpu-node-1") so config (role, pause, max_cameras) survives
# reconnects. Falls back to hostname if not set.
WORKER_NAME: str = os.getenv("WORKER_NAME", "") or socket.gethostname()


# ── Model sync from control plane S3 ─────────────────────────────────────────

def sync_models_from_control_plane(models_dir: str) -> bool:
    """
    Download model files from Go control plane's /api/models/ endpoints.
    Skips files that already exist locally with the same size.
    Returns True if any file was downloaded.
    """
    import urllib.request
    import json as _json

    http_url = os.environ.get("CONTROL_PLANE_HTTP_URL", "").rstrip("/")
    if not http_url:
        grpc_url = os.environ.get("CONTROL_PLANE_URL", "")
        if grpc_url:
            host = grpc_url.split(":")[0]
            http_url = f"http://{host}:8000"
    if not http_url:
        return False

    manifest_url = f"{http_url}/api/models/manifest"
    try:
        with urllib.request.urlopen(manifest_url, timeout=15) as r:
            manifest = _json.loads(r.read())
    except Exception as e:
        logger.warning(f"[ModelSync] Could not fetch manifest from {manifest_url}: {e}")
        return False

    files = manifest.get("files", [])
    if not files:
        logger.info("[ModelSync] Manifest is empty — no models in S3 yet")
        return False

    os.makedirs(models_dir, exist_ok=True)
    downloaded = False

    for fi in files:
        name = fi.get("name", "")
        if not name or name == "meta.json":
            continue

        local_path = os.path.join(models_dir, name)
        remote_size = fi.get("size", -1)

        # Skip if local file already has the same size
        if os.path.exists(local_path) and os.path.getsize(local_path) == remote_size:
            logger.debug(f"[ModelSync] {name} up-to-date ({remote_size} bytes) — skip")
            continue

        download_url = f"{http_url}/api/models/download/{name}"
        logger.info(f"[ModelSync] Downloading {name} ({remote_size / 1024 / 1024:.1f} MB)...")

        # Download to a temp file in the same dir, then atomically swap into place.
        # This prevents a failed/partial download from corrupting the live model
        # that the engine is about to load on reload.
        import tempfile
        tmp_fd, tmp_path = tempfile.mkstemp(dir=models_dir, suffix=".tmp")
        os.close(tmp_fd)
        try:
            urllib.request.urlretrieve(download_url, tmp_path)
            # Verify the download is complete before swapping
            if remote_size >= 0 and os.path.getsize(tmp_path) != remote_size:
                raise IOError(
                    f"size mismatch: got {os.path.getsize(tmp_path)}, expected {remote_size}"
                )
            os.replace(tmp_path, local_path)  # atomic on POSIX
            logger.info(f"[ModelSync] {name} saved to {local_path}")
            downloaded = True
        except Exception as e:
            logger.error(f"[ModelSync] Failed to download {name}: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    return downloaded

# FFHQ 5-point landmark template for 512×512.
# CodeFormer (and GFPGAN) were trained exclusively on FFHQ-aligned crops; using
# InsightFace's ArcFace template instead causes severe face distortion because the
# VQ-VAE codebook never saw that alignment during training.
_FFHQ_TEMPLATE_512 = np.array([
    [192.98138, 239.94708],
    [318.90277, 240.19360],
    [256.63416, 314.01935],
    [201.26117, 371.41043],
    [313.08905, 371.15118],
], dtype=np.float32)


def _align_face_ffhq(img: np.ndarray, kps: np.ndarray, output_size: int = 512) -> np.ndarray:
    """Warp face to FFHQ alignment at output_size × output_size."""
    dst = _FFHQ_TEMPLATE_512 * (output_size / 512.0)
    # LMEDS matches facexlib's FaceRestoreHelper — robust for exactly 5 landmarks
    # (RANSAC's reprojection threshold can spuriously reject points and skew the fit).
    M, _ = cv2.estimateAffinePartial2D(kps, dst, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(
        img, M, (output_size, output_size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("AI_Worker_gRPC")


class FaceTrack:
    """Represents a tracked face on a camera over a short buffering window."""

    def __init__(self, camera_id, bbox, embedding, task_id, image_bytes, sharpness: float = 0.0, frontality: float = 1.0, kps: list = None):
        self.camera_id = camera_id
        self.track_id = str(uuid.uuid4())
        self.bbox = bbox
        self.embedding = embedding
        self.task_id = task_id
        self.image_bytes = image_bytes
        self.face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        self.sharpness = sharpness
        self.frontality = frontality
        self.kps = kps
        self.quality_score = self.face_area * self.frontality
        
        self.first_seen = time.time()
        self.last_seen = time.time()

    def update(self, bbox, embedding, task_id, image_bytes, sharpness: float = 0.0, frontality: float = 1.0, kps: list = None):
        """Update the track's best frame using sharpness-gated + quality score selection.

        Quality score = face_area × frontality.
        This favors frames where the face is both large AND looking straight.

        Strategy:
        1. If the new frame is sharp and the current best is NOT sharp,
           always replace (sharp beats blurry regardless of quality).
        2. If both are sharp (or both are blurry), pick the one with
           the higher quality score (area × frontality).
        3. If the current best is sharp and the new frame is NOT,
           keep the current best (don't replace sharp with blurry).
        """
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        new_quality = area * frontality
        new_is_sharp = sharpness >= SHARPNESS_THRESHOLD
        cur_is_sharp = self.sharpness >= SHARPNESS_THRESHOLD

        should_replace = False
        if new_is_sharp and not cur_is_sharp:
            # New frame is sharp, current is blurry -> always replace
            should_replace = True
        elif new_is_sharp == cur_is_sharp:
            # Both sharp or both blurry -> pick higher quality score
            should_replace = new_quality > self.quality_score
        # else: current is sharp, new is blurry -> keep current

        if should_replace:
            self.bbox = bbox
            self.embedding = embedding
            self.task_id = task_id
            self.image_bytes = image_bytes
            self.face_area = area
            self.sharpness = sharpness
            self.frontality = frontality
            self.kps = kps
            self.quality_score = new_quality
        self.last_seen = time.time()


class PlateTrack:
    """Tracks a license plate bbox across frames and keeps the best OCR result."""

    def __init__(self, camera_id: int, plate_result: PlateResult, task_id: str):
        self.camera_id    = camera_id
        self.track_id     = str(uuid.uuid4())
        self.frame_results: list[PlateResult] = [plate_result]
        self.last_bbox    = plate_result.bbox
        self.task_id      = task_id
        self.hit_count    = 1
        self.first_seen   = time.time()
        self.last_seen    = time.time()

    def update(self, plate_result: PlateResult, task_id: str):
        """Accumulate every frame — best-char selection happens at flush time."""
        self.frame_results.append(plate_result)
        self.last_bbox = plate_result.bbox
        self.last_seen = time.time()
        self.task_id   = task_id
        self.hit_count += 1


def _iou(a: list[float], b: list[float]) -> float:
    """Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _plate_boxes_match(a: list[float], b: list[float]) -> bool:
    """Return True if two plate bboxes likely belong to the same physical plate.

    Pure IoU fails when the detection model returns slightly different crop sizes
    between frames (common with padded YOLO inference) or when the vehicle moves
    a small amount between 2fps captures.  We fall back to centre-distance so
    that any overlap OR proximity within ~80% of the plate width counts as a match.
    """
    if _iou(a, b) >= PLATE_IOU_THRESH:
        return True
    # Fallback: centres within 0.8× the average plate width
    cx_a = (a[0] + a[2]) / 2;  cy_a = (a[1] + a[3]) / 2
    cx_b = (b[0] + b[2]) / 2;  cy_b = (b[1] + b[3]) / 2
    avg_w = ((a[2] - a[0]) + (b[2] - b[0])) / 2
    dist  = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
    return dist < avg_w * 0.8


# Active tracks and cooldown state
active_tracks = {}
tracks_lock = threading.Lock()

cooldowns = []
cooldowns_lock = threading.Lock()

# Plate tracks
active_plate_tracks: dict[str, "PlateTrack"] = {}
plate_tracks_lock = threading.Lock()

plate_cooldowns: dict[str, float] = {}  # plate_number -> expires_at
plate_cooldowns_lock = threading.Lock()

# Camera state and logs suppression
last_seen_camera = {}
last_detect_mode = {}  # camera_id -> last logged detect_mode
cameras_lock = threading.Lock()

# Stats for average process time tracking
stats_lock = threading.Lock()
recent_process_times = []

TRACK_TIMEOUT = 3.0  # seconds of inactivity before flushing
TRACK_MAX_DURATION = 5.0  # max seconds a track can run before flushing
COOLDOWN_DURATION = 30.0  # seconds

# Inactivity window before a plate track flushes. Plate detections are sparse
# (a plate is only readable for a moment as a vehicle passes), so this needs to
# be wide enough that consecutive detections of the same plate land in one track
# and accumulate >= MIN_PLATE_HITS before timing out.
PLATE_TRACK_TIMEOUT = float(os.getenv("PLATE_TRACK_TIMEOUT", "6.0"))      # seconds of inactivity before flushing
PLATE_TRACK_MAX_DURATION = float(os.getenv("PLATE_TRACK_MAX_DURATION", "12.0"))
PLATE_COOLDOWN_DURATION = 10.0  # don't re-report same plate for 10 s
PLATE_IOU_THRESH = 0.4         # IoU threshold for matching same plate across frames
MIN_PLATE_HITS = int(os.getenv("MIN_PLATE_HITS", "1"))  # discard single-frame detections
# Frames with confidence AT OR ABOVE this threshold are captured for training review.
# Set to a medium value (not just very-high) so uncertain reads — where the model
# misses or misreads characters — flow into the pending queue. The reviewer can add
# missing boxes / fix classes in the UI, which is exactly the data that improves the
# model on its weak spots. Tune via TRAINING_CAPTURE_CONF_MIN env var.
TRAINING_CAPTURE_CONF_MIN = float(os.getenv("TRAINING_CAPTURE_CONF_MIN", "0.45"))
# Max frames per track to send as training data (pick highest-confidence ones)
TRAINING_MAX_FRAMES_PER_TRACK = int(os.getenv("TRAINING_MAX_FRAMES_PER_TRACK", "3"))
# When assembling a track's final read, vote over only the top-K highest-confidence
# frames instead of every frame. A track accumulates many blurry frames where the
# model misreads consonants as digits; including them drags the assembled read toward
# garbage (the digit-only plates seen on the all-plates page). Training already proves
# the few high-confidence frames read the plate correctly, so restrict the vote to that
# same cream. Set to 0 to disable (vote over all frames). Default mirrors training.
PLATE_ASSEMBLE_TOP_K = int(os.getenv("PLATE_ASSEMBLE_TOP_K", "5"))
# Multiplier applied to the stored confidence when a plate only became valid AFTER
# correct_common_errors() fixed an OCR artifact. A corrected read is genuinely less
# certain than one that matched plate rules directly, so its confidence should reflect
# that — otherwise corrected plates can outrank clean reads when filtering/sorting.
PLATE_CORRECTION_CONF_PENALTY = float(os.getenv("PLATE_CORRECTION_CONF_PENALTY", "0.8"))


def clean_cooldowns():
    global cooldowns
    now = time.time()
    with cooldowns_lock:
        cooldowns = [c for c in cooldowns if c["expires_at"] > now]


def is_on_cooldown(embedding):
    clean_cooldowns()
    with cooldowns_lock:
        for c in cooldowns:
            # Cosine similarity (dot product of normalized embeddings)
            sim = np.dot(embedding, c["embedding"])
            if sim > 0.6:
                return True
    return False


def add_cooldown(embedding):
    with cooldowns_lock:
        cooldowns.append({
            "embedding": embedding,
            "expires_at": time.time() + COOLDOWN_DURATION
        })


def clean_plate_cooldowns():
    now = time.time()
    with plate_cooldowns_lock:
        expired = [k for k, exp in plate_cooldowns.items() if exp <= now]
        for k in expired:
            del plate_cooldowns[k]


def is_plate_on_cooldown(plate_key: str) -> bool:
    now = time.time()
    with plate_cooldowns_lock:
        exp = plate_cooldowns.get(plate_key)
        return exp is not None and exp > now


def add_plate_cooldown(plate_number: str):
    with plate_cooldowns_lock:
        plate_cooldowns[_plate_cooldown_key(plate_number)] = time.time() + PLATE_COOLDOWN_DURATION




def _plate_cooldown_key(plate_number: str) -> str:
    """Normalize plate number for cooldown lookup.

    Strips hyphens and spaces so "7-ขว-1344" and "7ขว1344" share the same key.
    """
    import re as _re
    return _re.sub(r'[\s\-–—]', '', plate_number)


def _levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


# Max edit distance for fuzzy cooldown matching — covers 1-2 OCR misreads
PLATE_FUZZY_DIST = int(os.getenv("PLATE_FUZZY_DIST", "2"))


def claim_plate_slot(plate_number: str) -> bool:
    """Atomically check cooldown (exact + fuzzy) and claim the slot.

    Returns True if the caller should proceed (not on cooldown).
    Fuzzy match blocks OCR variants of the same physical plate
    (e.g. ญพ8378 vs ขพ8378 differ by 1 char → same cooldown slot).
    """
    key = _plate_cooldown_key(plate_number)
    now = time.time()
    with plate_cooldowns_lock:
        # Fast path: exact match
        exp = plate_cooldowns.get(key)
        if exp is not None and exp > now:
            return False
        # Fuzzy path: scan active cooldowns for near-matches
        for existing_key, exp in plate_cooldowns.items():
            if exp <= now:
                continue
            if _levenshtein(key, existing_key) <= PLATE_FUZZY_DIST:
                return False
        plate_cooldowns[key] = now + PLATE_COOLDOWN_DURATION
        return True


def _assemble_multi_frame(frame_results: list) -> PlateResult:
    """Assemble the best plate reading from all frames in a track.

    Algorithm:
      1. Group frames by character count.
      2. Pick the majority count; fallback to best-confidence frame if no majority.
      3. For each character position (slot), pick the char with highest confidence
         across all majority frames.
      4. Validate and return a new PlateResult whose confidence is the mean of
         per-slot best confidences (no single weak character drags it down).
    """
    # Step 0 — drop structurally-implausible frames BEFORE voting.
    # The detector sometimes emits garbage frames (all-digit reads, or absurd char
    # counts like 9) where it mistook consonants for numbers. If those pollute the
    # vote they produce an assembled read that is digit-only and worse than any single
    # clean frame — exactly the '377131565' / '986689' cases seen in production. A real
    # Thai plate always has >= 1 consonant and is never longer than ~8 chars, so:
    #   1. keep only frames with a plausible char count (2..8)
    #   2. if ANY surviving frame actually saw a consonant, discard the consonant-less
    #      (pure-digit) frames entirely — they are detector noise, not partial reads.
    def _has_consonant(pr) -> bool:
        return any('ก' <= c.char <= 'ฮ' for c in pr.chars)

    usable = [pr for pr in frame_results if pr.chars and 2 <= len(pr.chars) <= 8]
    with_cons = [pr for pr in usable if _has_consonant(pr)]
    if with_cons:
        usable = with_cons

    # Fallback: nothing plausible to assemble (old-path or engine error)
    if not usable:
        return max(frame_results, key=lambda pr: pr.confidence)

    # Step 0.5 — keep only the top-K highest-confidence frames before voting.
    # Voting over the whole track lets many low-confidence (blurry/angled) frames —
    # where consonants misread as digits — dominate the count-group choice and the
    # per-slot vote, producing digit-only assembled reads. Restricting the vote to the
    # same high-confidence cream that training captures keeps the assembled read aligned
    # with what the model actually reads well. Consonant-bearing frames are already
    # preferred above, so this never drops the only consonant evidence in favour of digits.
    if PLATE_ASSEMBLE_TOP_K > 0 and len(usable) > PLATE_ASSEMBLE_TOP_K:
        usable = sorted(usable, key=lambda pr: pr.confidence, reverse=True)[:PLATE_ASSEMBLE_TOP_K]

    # Step 1 — group surviving frames by char count
    count_groups: dict = defaultdict(list)
    for pr in usable:
        count_groups[len(pr.chars)].append(pr)

    # Step 2 — pick the count group with the highest TOTAL confidence, not the most
    # frames. A few high-confidence reads of the right length should outweigh many
    # low-confidence garbage frames that happen to share a wrong length.
    best_count = max(count_groups, key=lambda k: sum(p.confidence for p in count_groups[k]))
    candidates = count_groups[best_count]

    # Step 3 — slot pool: slot_idx → list of (char, conf)
    slot_pool: dict = defaultdict(list)
    for pr in candidates:
        for ch in pr.chars:
            slot_idx = min(int(ch.x_norm * best_count), best_count - 1)
            slot_pool[slot_idx].append((ch.char, ch.confidence))

    # If any slot is empty (shouldn't happen with count filter) → fallback
    if len(slot_pool) < best_count:
        return max(frame_results, key=lambda pr: pr.confidence)

    assembled_chars = []
    total_conf = 0.0
    for i in range(best_count):
        best_char, best_conf = max(slot_pool[i], key=lambda x: x[1])
        assembled_chars.append(best_char)
        total_conf += best_conf

    raw_text = "".join(assembled_chars)
    mean_conf = total_conf / best_count

    # Step 4 — validate against Thai plate rules
    import re as _re
    was_corrected = False
    is_valid, normalized, _ = LicensePlateValidator.validate(raw_text)
    if not is_valid:
        corrected = LicensePlateValidator.correct_common_errors(raw_text)
        if corrected:
            is_valid, normalized, _ = LicensePlateValidator.validate(corrected)
            # Only count it as a "correction" if actual characters changed (an OCR
            # artifact fix like เ→4 or B→8). Plain hyphen/space insertion is cosmetic
            # — validate() requires a hyphen, so every clean read passes through here —
            # and must NOT be penalized.
            if is_valid:
                strip = lambda s: _re.sub(r'[\s\-–—]', '', s)
                was_corrected = strip(corrected) != strip(raw_text)

    plate_number = normalized if is_valid else None

    # Confidence policy:
    #   no valid plate        → halve (noisy / unparseable)
    #   plate via correction  → penalty: the raw read didn't satisfy plate rules on its
    #                           own, so the stored confidence shouldn't look as trustworthy
    #                           as a clean read
    #   plate read cleanly    → full mean confidence
    if plate_number is None:
        final_conf = mean_conf * 0.5
    elif was_corrected:
        final_conf = mean_conf * PLATE_CORRECTION_CONF_PENALTY
    else:
        final_conf = mean_conf

    # Province: majority vote across candidate frames
    provinces = [pr.province for pr in candidates if pr.province]
    province = Counter(provinces).most_common(1)[0][0] if provinces else None

    best_frame = max(candidates, key=lambda pr: pr.confidence)

    # Snapshot geometry MUST come from the LAST frame, not the best-confidence frame.
    # The Go side crops the wide vehicle snapshot from the image of `track.task_id`, and
    # PlateTrack.update() always advances task_id together with appending to frame_results
    # — so `track.task_id` == the last frame (frame_results[-1]). If we sent the best
    # frame's bbox here, Go would crop that bbox out of a *different* frame's image; the
    # car has moved between the two frames, so the crop lands off the plate ("คนละเฟรมกัน").
    # OCR text/province still use the best frames above; only the snapshot bbox + crop are
    # pinned to the last frame so they stay aligned with the image Go actually has.
    snap_frame = frame_results[-1]

    return PlateResult(
        plate_number=plate_number,
        confidence=final_conf,
        bbox=snap_frame.bbox,                 # aligned with track.task_id's image (last frame)
        plate_type=best_frame.plate_type,
        province=province,
        raw_text=raw_text,
        chars=[],
        crop_bytes=snap_frame.crop_bytes,     # deskewed crop from the same (last) frame
    )


def flush_plate_track(track: PlateTrack, send_queue):
    if track.hit_count < MIN_PLATE_HITS:
        raw_sample = track.frame_results[0].raw_text if track.frame_results else "?"
        logger.info(
            f"Discarding plate track for camera {track.camera_id} "
            f"(hits={track.hit_count} < {MIN_PLATE_HITS}) — single-frame detection  raw='{raw_sample}'"
        )
        return

    pr = _assemble_multi_frame(track.frame_results)
    label = pr.plate_number or pr.raw_text or "?"

    # Gate: only flush reads that parsed into a valid Thai plate. An unparseable read
    # (plate_number is None) is OCR noise — a real plate always has >=2 consonants + a
    # number block, so a read that fails validation (pure-digit junk like '3764255',
    # partial reads like '22ก', wrong-length garbage) is never a real plate and must not
    # pollute the detection log regardless of its raw confidence. Training-frame capture
    # below is independent of this gate, so dropping these costs no training data.
    if pr.plate_number is None:
        logger.info(
            f"Discarding plate track for camera {track.camera_id} "
            f"(no valid plate number; conf={pr.confidence:.2f}) raw='{pr.raw_text}'"
        )
        return

    if pr.plate_number:
        if not claim_plate_slot(pr.plate_number):
            logger.info(f"Plate {pr.plate_number} on cooldown — skipping flush")
            return

    logger.info(
        f"Flushing plate track for camera {track.camera_id}: "
        f"{label}  conf={pr.confidence:.2f}  hits={track.hit_count}"
    )

    import json as _json

    def _char_labels_json(chars) -> str:
        return _json.dumps([
            {"class_name": c.char, "cx": c.cx, "cy": c.cy,
             "bw": c.bw, "bh": c.bh, "confidence": c.confidence}
            for c in chars
        ], ensure_ascii=False)

    # Collect high-confidence frames as auto-labeled training data (cap at N highest-conf)
    candidates = [
        fr for fr in track.frame_results
        if fr.confidence >= TRAINING_CAPTURE_CONF_MIN and fr.chars and fr.crop_bytes
    ]
    candidates.sort(key=lambda fr: fr.confidence, reverse=True)
    training_frames = [
        facerec_pb2.PlateTrainingFrame(
            crop_jpeg=fr.crop_bytes,
            char_labels_json=_char_labels_json(fr.chars),
            confidence=fr.confidence,
            raw_text=fr.raw_text,
            track_id=track.track_id,
        )
        for fr in candidates[:TRAINING_MAX_FRAMES_PER_TRACK]
    ]

    send_queue.put(facerec_pb2.InferenceResult(
        task_id=track.task_id,
        detections=[],
        plate_detections=[facerec_pb2.PlateDetection(
            bbox=pr.bbox,
            plate_number=pr.plate_number or "",
            confidence=pr.confidence,
            plate_type=pr.plate_type,
            province=pr.province or "",
            raw_text=pr.raw_text,
            char_labels_json=_char_labels_json(pr.chars) if pr.chars else "",
            snapshot_jpeg=pr.crop_bytes or b"",
        )],
        plate_training_frames=training_frames,
    ))


def flush_track(track, send_queue):
    # Discard track if the best frame is still extremely blurry
    min_track_sharpness = float(os.getenv("MIN_TRACK_SHARPNESS", "30.0"))
    if track.sharpness < min_track_sharpness:
        logger.info(f"Discarding track {track.track_id} on camera {track.camera_id} because best frame sharpness ({track.sharpness:.1f}) is below minimum threshold ({min_track_sharpness:.1f})")
        return

    logger.info(f"Flushing best face track for camera {track.camera_id} (Area: {track.face_area:.0f}, Sharpness: {track.sharpness:.1f}, Frontality: {track.frontality:.2f}, Quality: {track.quality_score:.0f})")
    add_cooldown(track.embedding)
    
    restored_face_bytes = b""
    if face_restorer.is_enabled():
        try:
            # Landmarks are required: CodeFormer only works on FFHQ-aligned faces.
            # Without kps we can't align, and feeding an unaligned crop produces
            # severe distortion — so skip restoration entirely in that case.
            kps = getattr(track, 'kps', None)
            if kps is None:
                logger.debug(f"Skipping restoration for track {track.track_id}: no landmarks")
            else:
                kps_arr = np.array(kps, dtype=np.float32)
                if kps_arr.shape == (10,):
                    kps_arr = kps_arr.reshape(5, 2)
                if kps_arr.shape != (5, 2):
                    logger.warning(f"Skipping restoration: unexpected kps shape {kps_arr.shape}")
                else:
                    nparr = np.frombuffer(track.image_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        # FFHQ-align directly to 512 — CodeFormer's native input size and
                        # training distribution. It is a generative restorer that performs
                        # its own super-resolution internally, so no upscaler pre-pass is needed.
                        face_crop = _align_face_ffhq(img, kps_arr, output_size=512)
                        if face_crop is not None and face_crop.size > 0:
                            # Quality gate: CodeFormer helps blurry/small faces but
                            # over-smooths faces that are already sharp AND large,
                            # shifting their appearance. Skip it for good-quality faces
                            # and ship the plain FFHQ-aligned crop instead. This also
                            # cuts contention on the global GPU inference lock.
                            sharpness_max = float(os.getenv("CODEFORMER_RESTORE_SHARPNESS_MAX", "120.0"))
                            area_max = float(os.getenv("CODEFORMER_RESTORE_AREA_MAX", "40000.0"))
                            already_good = track.sharpness >= sharpness_max and track.face_area >= area_max
                            if already_good:
                                restored = None
                                logger.debug(
                                    f"Skipping CodeFormer for track {track.track_id}: face already "
                                    f"sharp+large (sharpness {track.sharpness:.0f}, area {track.face_area:.0f})"
                                )
                            else:
                                restored = face_restorer.restore_face(face_crop)
                            final_img = restored if restored is not None else face_crop
                            success, encoded_img = cv2.imencode(".jpg", final_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            if success:
                                restored_face_bytes = encoded_img.tobytes()
                                stages = "CodeFormer" if restored is not None else "aligned crop"
                                logger.info(f"Restored face for track {track.track_id} via {stages}")
        except Exception as e:
            logger.error(f"Error during face restoration in flusher: {e}", exc_info=True)

    result = facerec_pb2.InferenceResult(
        task_id=track.task_id,
        detections=[facerec_pb2.Detection(
            bbox=track.bbox,
            embedding=track.embedding.tolist(),
            restored_face_jpeg=restored_face_bytes
        )]
    )
    send_queue.put(result)


def track_flusher(send_queue, stop_event=None):
    logger.info("Background Face Track Flusher thread started.")
    last_stats_sent = 0
    max_workers = int(os.getenv("AI_WORKER_CONCURRENCY", "4"))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            if stop_event and stop_event.is_set():
                logger.info("Flusher stopping due to stop event.")
                break
            try:
                time.sleep(0.5)
                now = time.time()
                to_flush = []
                
                with tracks_lock:
                    keys = list(active_tracks.keys())
                    for key in keys:
                        track = active_tracks[key]
                        if (now - track.last_seen > TRACK_TIMEOUT) or (now - track.first_seen > TRACK_MAX_DURATION):
                            to_flush.append(track)
                            del active_tracks[key]

                for track in to_flush:
                    executor.submit(flush_track, track, send_queue)

                # Flush expired plate tracks
                to_flush_plates = []
                with plate_tracks_lock:
                    for key in list(active_plate_tracks.keys()):
                        pt = active_plate_tracks[key]
                        if (now - pt.last_seen > PLATE_TRACK_TIMEOUT) or (now - pt.first_seen > PLATE_TRACK_MAX_DURATION):
                            to_flush_plates.append(pt)
                            del active_plate_tracks[key]

                for pt in to_flush_plates:
                    executor.submit(flush_plate_track, pt, send_queue)

                clean_plate_cooldowns()

                # Periodically send metrics update (every 2 seconds)
                if now - last_stats_sent > 2.0:
                    last_stats_sent = now
                    with stats_lock:
                        if recent_process_times:
                            avg_ms = sum(recent_process_times) / len(recent_process_times)
                            result = facerec_pb2.InferenceResult(
                                task_id="metrics",
                                detections=[],
                                process_time_ms=avg_ms
                            )
                            send_queue.put(result)

                # Clean up inactive cameras to log stop events
                inactive_timeout = 10.0
                with cameras_lock:
                    for cam_id, last_ts in list(last_seen_camera.items()):
                        if now - last_ts > inactive_timeout:
                            logger.info(f"Stopped processing stream for Camera {cam_id} (inactive)")
                            del last_seen_camera[cam_id]
                            last_detect_mode.pop(cam_id, None)
            except Exception as e:
                logger.error(f"Error in track_flusher: {e}")


def _rewrite_dataset_yaml(yaml_path: str, send_queue) -> str:
    """
    Rewrite a YOLO data.yaml so train/val point at the real directory the dataset
    was extracted into on this machine. The control plane writes absolute paths
    from its own temp dir, which are invalid on the worker. We derive the correct
    paths from the yaml's own location (zip root) and the standard YOLO layout.
    """
    import yaml as _yaml

    root = os.path.dirname(os.path.abspath(yaml_path))
    try:
        with open(yaml_path, encoding="utf-8") as f:
            cfg = _yaml.safe_load(f) or {}

        def resolve_split(*candidates):
            for sub in candidates:
                cand = os.path.join(root, sub, "images")
                if os.path.isdir(cand):
                    return cand
            return None

        train_dir = resolve_split("train")
        val_dir = resolve_split("valid", "val")

        # Use an absolute `path` root + relative splits so ultralytics resolves
        # against the dataset dir, not its global datasets_dir setting.
        cfg["path"] = root
        if train_dir:
            cfg["train"] = os.path.relpath(train_dir, root)
        if val_dir:
            cfg["val"] = os.path.relpath(val_dir, root)
            cfg.pop("valid", None)

        with open(yaml_path, "w", encoding="utf-8") as f:
            _yaml.safe_dump(cfg, f, default_flow_style=False, allow_unicode=True)

        _send_finetune_progress(send_queue, type="info",
                                message=f"Rewrote data.yaml — path={root} train={cfg.get('train')} val={cfg.get('val')}")
    except Exception as e:
        _send_finetune_progress(send_queue, type="info",
                                message=f"Warning: could not rewrite data.yaml: {e}")
    return yaml_path


def _upload_model_file(http_url: str, version: str, file_path: str, send_queue) -> bool:
    """Upload a trained model file to the control plane via raw PUT body. Returns True on success."""
    import urllib.request

    filename = os.path.basename(file_path)
    url = f"{http_url}/api/models/upload/{version}/{filename}"
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        req = urllib.request.Request(
            url, data=data, method="PUT",
            headers={"Content-Type": "application/octet-stream"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            resp.read()
        _send_finetune_progress(send_queue, type="info",
                                message=f"Uploaded {filename} ({len(data)} bytes) to control plane")
        return True
    except Exception as e:
        _send_finetune_progress(send_queue, type="info",
                                message=f"Upload {filename} failed: {e}")
        return False


_finetune_stop_event = threading.Event()


def _send_finetune_progress(send_queue, **kwargs):
    """Helper to send FinetuneProgress back to the control plane via gRPC."""
    fp = facerec_pb2.FinetuneProgress(**kwargs)
    result = facerec_pb2.InferenceResult(
        task_id="finetune",
        finetune_progress=fp,
    )
    try:
        send_queue.put(result, timeout=10)
    except Exception:
        pass


def _run_finetune(s3_key: str, epochs: int, send_queue, roboflow_api_key: str = ""):
    """Download the CCTV dataset zip from S3, merge with Roboflow datasets, run fine-tuning."""
    import zipfile
    import tempfile
    import shutil

    _finetune_stop_event.clear()
    _send_finetune_progress(send_queue, type="info", message="Worker received finetune task — preparing dataset")

    # Download dataset zip from S3 (via control plane HTTP)
    http_url = os.environ.get("CONTROL_PLANE_HTTP_URL", "").rstrip("/")
    if not http_url:
        grpc_url = os.environ.get("CONTROL_PLANE_URL", "")
        if grpc_url:
            host = grpc_url.split(":")[0]
            http_url = f"http://{host}:8000"

    cctv_yaml = None
    tmp_dataset_dir = None

    if s3_key and http_url:
        import urllib.request
        try:
            zip_url = f"{http_url}/api/static/snapshots/{s3_key}"
            _send_finetune_progress(send_queue, type="info", message=f"Downloading dataset: {zip_url}")
            tmp_dataset_dir = tempfile.mkdtemp(prefix="finetune_ds_")
            zip_path = os.path.join(tmp_dataset_dir, "cctv_dataset.zip")
            urllib.request.urlretrieve(zip_url, zip_path)
            extract_dir = os.path.join(tmp_dataset_dir, "cctv")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            os.remove(zip_path)
            for root, _, files in os.walk(extract_dir):
                for fn in files:
                    if fn == "data.yaml":
                        cctv_yaml = os.path.join(root, fn)
                        break
                if cctv_yaml:
                    break
            _send_finetune_progress(send_queue, type="info", message=f"Dataset extracted — data.yaml: {cctv_yaml}")

            # The data.yaml written by the control plane carries absolute paths
            # from ITS temp dir, which don't exist here. Rewrite train/val to point
            # at the actual extracted location on this worker.
            if cctv_yaml:
                cctv_yaml = _rewrite_dataset_yaml(cctv_yaml, send_queue)
        except Exception as e:
            _send_finetune_progress(send_queue, type="info",
                                    message=f"Warning: could not download CCTV dataset: {e}. Training with Roboflow datasets only.")

    # Locate models dir
    from app.license_plate.engine import _resolve_models_dir
    models_dir = _resolve_models_dir()
    base_model = os.path.join(models_dir, "thai_char_yolo26s.pt")
    output_model = base_model

    roboflow_base = os.environ.get("ROBOFLOW_DATASET_BASE", "")
    if not roboflow_base:
        roboflow_base = os.path.join(os.path.dirname(models_dir), "datasets")
    os.makedirs(roboflow_base, exist_ok=True)

    def _on_progress(msg: dict):
        msg_type = msg.get("type", "info")
        if msg_type == "epoch":
            _send_finetune_progress(send_queue,
                type="epoch",
                epoch=int(msg.get("epoch", 0)),
                epochs=int(msg.get("epochs", epochs)),
                box_loss=float(msg.get("box_loss", 0)),
                cls_loss=float(msg.get("cls_loss", 0)))
        elif msg_type == "done":
            version = msg.get("version", "")
            # Trained model lives on THIS worker's disk. Upload it back to the
            # control plane before signalling "done" so the control plane can
            # version, push to S3, and reload all workers.
            if version and http_url:
                ok = True
                for fpath in (msg.get("model", ""), msg.get("onnx", "")):
                    if fpath and os.path.exists(fpath):
                        if not _upload_model_file(http_url, version, fpath, send_queue):
                            ok = False
                if not ok:
                    _send_finetune_progress(send_queue, type="error",
                                            message="Model upload to control plane failed — not activating")
                    return
            elif version and not http_url:
                _send_finetune_progress(send_queue, type="error",
                                        message="No control plane HTTP URL — cannot upload trained model")
                return
            _send_finetune_progress(send_queue, type="done",
                                    version=version,
                                    message="Training complete")
        elif msg_type == "error":
            _send_finetune_progress(send_queue, type="error",
                                    message=msg.get("message", str(msg)))
        else:
            _send_finetune_progress(send_queue, type="info",
                                    message=msg.get("message", str(msg)))

    try:
        import finetune_char_model as _ftm
        _send_finetune_progress(send_queue, type="info",
                                message=f"Starting training (inline) — base={base_model} epochs={epochs}")
        _ftm.run_finetune_inline(
            base_model=base_model,
            output_model=output_model,
            epochs=epochs,
            cctv_yaml=cctv_yaml,
            roboflow_base=roboflow_base,
            progress_cb=_on_progress,
            stop_event=_finetune_stop_event,
            roboflow_api_key=roboflow_api_key,
        )
    except Exception as e:
        _send_finetune_progress(send_queue, type="error", message=f"Training failed: {e}")
    finally:
        if tmp_dataset_dir:
            try:
                shutil.rmtree(tmp_dataset_dir, ignore_errors=True)
            except Exception:
                pass


def process_task(task_id, image_data, is_reg, send_queue, detect_mode="face"):
    try:
        start_time = time.time()
        # Decode image from JPEG bytes
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error(f"Failed to decode JPEG image for task {task_id}")
            if is_reg:
                duration_ms = (time.time() - start_time) * 1000
                send_queue.put(facerec_pb2.InferenceResult(task_id=task_id, detections=[], process_time_ms=duration_ms))
            return
            
        if is_reg:
            # Registration mode: extract embedding and return immediately
            emb, err_msg = face_engine.extract_embedding_from_image(img)
            duration_ms = (time.time() - start_time) * 1000
            if err_msg is not None:
                logger.warning(f"Registration face rejected: {err_msg}")
                result = facerec_pb2.InferenceResult(
                    task_id=task_id,
                    detections=[],
                    error_message=err_msg,
                    process_time_ms=duration_ms
                )
                send_queue.put(result)
            elif emb is not None:
                result = facerec_pb2.InferenceResult(
                    task_id=task_id,
                    detections=[facerec_pb2.Detection(
                        bbox=[0.0, 0.0, 0.0, 0.0],
                        embedding=emb.tolist()
                    )],
                    process_time_ms=duration_ms
                )
                send_queue.put(result)
            else:
                logger.warning(f"No face detected in registration image for task {task_id}")
                send_queue.put(facerec_pb2.InferenceResult(task_id=task_id, detections=[], error_message="No face detected", process_time_ms=duration_ms))
        else:
            # Parse camera_id from task_id (cameraID_timestamp_uuid)
            parts = task_id.split('_')
            camera_id = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0

            now_time = time.time()
            with cameras_lock:
                if camera_id not in last_seen_camera:
                    logger.info(f"Started processing stream for Camera {camera_id} (detect_mode='{detect_mode}')")
                elif last_detect_mode.get(camera_id) != detect_mode:
                    logger.info(f"Camera {camera_id} detect_mode changed: '{last_detect_mode.get(camera_id)}' -> '{detect_mode}'")
                last_detect_mode[camera_id] = detect_mode
                last_seen_camera[camera_id] = now_time

            faces = []

            # Run plate detection when mode is "plate" or "both"
            if detect_mode not in ("plate", "both") and license_plate_engine is not None and license_plate_engine.ready:
                logger.debug(f"Camera {camera_id}: skipping plate detection (detect_mode='{detect_mode}')")
            if detect_mode in ("plate", "both") and license_plate_engine is not None and license_plate_engine.ready:
                plate_results: list[PlateResult] = license_plate_engine.detect(img)
                if plate_results:
                    logger.info(f"Camera {camera_id}: detected {len(plate_results)} plate(s) this frame raw={[p.raw_text for p in plate_results]}")
                with plate_tracks_lock:
                    for pr in plate_results:
                        matched_pt = None
                        for pt in active_plate_tracks.values():
                            if pt.camera_id == camera_id and _plate_boxes_match(pt.last_bbox, pr.bbox):
                                matched_pt = pt
                                break
                        if matched_pt:
                            matched_pt.update(pr, task_id)
                        else:
                            new_pt = PlateTrack(camera_id, pr, task_id)
                            active_plate_tracks[new_pt.track_id] = new_pt

            # Run face detection when mode is "face" or "both"
            if detect_mode in ("face", "both"):
                faces = face_engine.detect_faces(img)

            duration_ms = (time.time() - start_time) * 1000
            if faces:
                logger.info(f"Processed frame for Camera {camera_id} in {duration_ms:.1f}ms - detected {len(faces)} face(s)")

            with stats_lock:
                recent_process_times.append(duration_ms)
                if len(recent_process_times) > 50:
                    recent_process_times.pop(0)

            # Plate-only mode: no face tracks to manage, just return
            if detect_mode == "plate":
                return

            with tracks_lock:
                for face in faces:
                    emb = face.embedding
                    bbox = face.bbox
                    sharpness = face.sharpness
                    frontality = face.frontality

                    # Check if on cooldown
                    if is_on_cooldown(emb):
                        continue

                    # Try to associate with active tracks for this camera
                    matched_track = None
                    for track in active_tracks.values():
                        if track.camera_id == camera_id:
                            sim = np.dot(emb, track.embedding)
                            if sim > 0.6:
                                matched_track = track
                                break

                    if matched_track:
                        matched_track.update(bbox, emb, task_id, image_data, sharpness, frontality, face.kps)
                    else:
                        # Create new track
                        new_track = FaceTrack(camera_id, bbox, emb, task_id, image_data, sharpness, frontality, face.kps)
                        active_tracks[new_track.track_id] = new_track
                        
    except Exception as ex:
        logger.error(f"Error processing task {task_id}: {ex}")
        if is_reg:
            send_queue.put(facerec_pb2.InferenceResult(task_id=task_id, detections=[]))


def run_grpc_client(control_plane_url=None, onnx_provider=None, stop_event=None):
    global license_plate_engine

    if control_plane_url is None:
        control_plane_url = os.getenv("CONTROL_PLANE_URL", "localhost:50051")

    if onnx_provider:
        os.environ["ONNX_PROVIDER"] = onnx_provider

    # Initialize InsightFace model
    logger.info("Initializing Face Engine...")
    face_engine.initialize()
    logger.info("Face Engine initialized successfully.")

    # Initialize CodeFormer Face Restorer
    logger.info("Initializing Face Restorer (CodeFormer)...")
    face_restorer.initialize()
    if face_restorer.is_enabled():
        logger.info("Face Restorer initialized successfully.")
    else:
        logger.warning("Face Restorer disabled (model not found or load failed).")

    # Sync models from S3 via control plane before loading
    from app.license_plate.engine import _resolve_models_dir
    _models_dir = _resolve_models_dir()
    logger.info(f"[ModelSync] Syncing models from control plane to {_models_dir} ...")
    sync_models_from_control_plane(_models_dir)

    # Initialize License Plate Engine
    logger.info("Initializing License Plate Engine...")
    license_plate_engine = LicensePlateEngine()
    if license_plate_engine.ready:
        logger.info("License Plate Engine initialized successfully.")
    else:
        logger.warning("License Plate Engine disabled (models not found).")

    def sleep_interruptible(seconds):
        steps = int(seconds / 0.2)
        for _ in range(steps):
            if stop_event and stop_event.is_set():
                return
            time.sleep(0.2)

    flusher_stop = None  # Tracks the current flusher's stop event for cleanup on reconnect

    while True:
        if stop_event and stop_event.is_set():
            logger.info("Stop event detected in main loop. Exiting client.")
            break

        # Stop the previous flusher before starting a new connection, so threads
        # don't accumulate across reconnects. Also clear stale per-connection state.
        if flusher_stop is not None:
            flusher_stop.set()

        with tracks_lock:
            active_tracks.clear()
        with plate_tracks_lock:
            active_plate_tracks.clear()
        with cameras_lock:
            last_seen_camera.clear()
            last_detect_mode.clear()

        logger.info(f"Connecting to gRPC server at {control_plane_url}...")
        try:
            # Configure gRPC channel options for large messages (up to 20MB)
            options = [
                ('grpc.max_receive_message_length', 20 * 1024 * 1024),
                ('grpc.max_send_message_length', 20 * 1024 * 1024),
            ]

            with grpc.insecure_channel(control_plane_url, options=options) as channel:
                stub = facerec_pb2_grpc.FaceInferenceServiceStub(channel)

                # Thread-safe queue for sending inference results back to server
                send_queue = queue.Queue()

                # Per-connection stop event so we can cleanly stop this flusher on
                # the next reconnect without touching the caller's stop_event.
                flusher_stop = threading.Event()

                # Start track flusher thread
                flusher_thread = threading.Thread(target=track_flusher, args=(send_queue, flusher_stop), daemon=True)
                flusher_thread.start()

                def request_generator():
                    # Identify this worker to the server before any real messages.
                    yield facerec_pb2.InferenceResult(task_id="hello", worker_name=WORKER_NAME)
                    logger.info(f"Sent worker identification: {WORKER_NAME}")

                    while True:
                        if stop_event and stop_event.is_set():
                            send_queue.put(None)
                            break
                        try:
                            item = send_queue.get(timeout=0.5)
                            if item is None:
                                break
                            if getattr(item, 'task_id', '') != 'metrics':
                                logger.debug(f"Yielding task {item.task_id} to gRPC stream (detections: {len(item.detections)})")
                            # Attach worker identity to every message for traceability
                            item.worker_name = WORKER_NAME
                            yield item
                        except queue.Empty:
                            continue

                # Start the bidirectional stream
                responses = stub.ProcessStream(request_generator())
                logger.info("gRPC stream established. Listening for tasks...")

                max_workers = int(os.getenv("AI_WORKER_CONCURRENCY", "4"))
                logger.info(f"Starting ThreadPoolExecutor with {max_workers} workers for bulk processing.")

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for task in responses:
                        if stop_event and stop_event.is_set():
                            break

                        # Hot-reload: sync latest model from S3 then reinitialise engine
                        if getattr(task, 'reload_models', False):
                            logger.info("Received reload_models signal — syncing from S3 and reinitialising LicensePlateEngine")
                            try:
                                from app.license_plate.engine import _resolve_models_dir
                                sync_models_from_control_plane(_resolve_models_dir())
                                license_plate_engine = LicensePlateEngine()
                                logger.info("LicensePlateEngine reloaded successfully")
                            except Exception as reload_err:
                                logger.error(f"Failed to reload engine: {reload_err}")
                            continue

                        # Stop finetune signal
                        if getattr(task, 'stop_finetune', False):
                            logger.info("Received stop_finetune signal — setting stop event")
                            _finetune_stop_event.set()
                            continue

                        # Fine-tune signal: run training in background thread
                        if getattr(task, 'start_finetune', False):
                            s3_key = getattr(task, 'finetune_dataset_s3_key', '')
                            epochs = getattr(task, 'finetune_epochs', 30) or 30
                            roboflow_api_key = getattr(task, 'roboflow_api_key', '') or ''
                            logger.info(f"Received start_finetune signal — dataset_s3_key={s3_key} epochs={epochs}")
                            executor.submit(_run_finetune, s3_key, epochs, send_queue, roboflow_api_key)
                            continue

                        task_id = task.task_id
                        image_data = task.image_data
                        is_reg = task.is_registration
                        detect_mode = getattr(task, 'detect_mode', None) or "face"

                        logger.debug(f"Received task: id={task_id}, size={len(image_data)} bytes, is_registration={is_reg}, detect_mode={detect_mode}")

                        # Submit task to thread pool
                        executor.submit(process_task, task_id, image_data, is_reg, send_queue, detect_mode)

        except grpc.RpcError as e:
            logger.error(f"gRPC stream connection error: {e.details() if hasattr(e, 'details') else e}. Retrying in 5 seconds...")
            sleep_interruptible(5)
        except Exception as e:
            logger.error(f"Unexpected error: {e}. Retrying in 5 seconds...")
            sleep_interruptible(5)


if __name__ == "__main__":
    run_grpc_client()
