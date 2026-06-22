from __future__ import annotations

import logging
import os
import queue
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
from app.face_engine import face_engine, compute_sharpness, SHARPNESS_THRESHOLD
from app.face_restorer import face_restorer
from app.license_plate import LicensePlateEngine, PlateResult

license_plate_engine: LicensePlateEngine | None = None

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
        self.camera_id  = camera_id
        self.track_id   = str(uuid.uuid4())
        self.best       = plate_result
        # Most recent bbox — used for spatial (IoU) matching. Kept separate
        # from best.bbox because a moving plate shifts position every frame
        # while best stays frozen on the best-OCR frame.
        self.last_bbox  = plate_result.bbox
        self.task_id    = task_id
        self.hit_count  = 1
        self.first_seen = time.time()
        self.last_seen  = time.time()

    def update(self, plate_result: PlateResult, task_id: str):
        """
        Keep the best OCR result:
          1. Valid plate_number beats no plate_number.
          2. Among equals, higher confidence wins.
        """
        current_valid = self.best.plate_number is not None
        new_valid     = plate_result.plate_number is not None

        if new_valid and not current_valid:
            self.best    = plate_result
            self.task_id = task_id
        elif new_valid == current_valid and plate_result.confidence > self.best.confidence:
            self.best    = plate_result
            self.task_id = task_id

        # Follow the plate's current position regardless of OCR quality.
        self.last_bbox = plate_result.bbox
        self.last_seen = time.time()
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
PLATE_COOLDOWN_DURATION = 10.0 # don't re-report same plate for 10 s
PLATE_IOU_THRESH = 0.4         # IoU threshold for matching same plate across frames
MIN_PLATE_HITS = int(os.getenv("MIN_PLATE_HITS", "2"))  # discard single-frame detections


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


def is_plate_on_cooldown(plate_number: str) -> bool:
    now = time.time()
    with plate_cooldowns_lock:
        exp = plate_cooldowns.get(plate_number)
        return exp is not None and exp > now


def add_plate_cooldown(plate_number: str):
    with plate_cooldowns_lock:
        plate_cooldowns[plate_number] = time.time() + PLATE_COOLDOWN_DURATION


def flush_plate_track(track: PlateTrack, send_queue):
    if track.hit_count < MIN_PLATE_HITS:
        logger.info(
            f"Discarding plate track for camera {track.camera_id} "
            f"(hits={track.hit_count} < {MIN_PLATE_HITS}) — single-frame detection  raw='{track.best.raw_text}'"
        )
        return

    pr = track.best
    label = pr.plate_number or pr.raw_text or "?"

    if pr.plate_number and is_plate_on_cooldown(pr.plate_number):
        logger.info(f"Plate {pr.plate_number} on cooldown — skipping flush")
        return

    logger.info(
        f"Flushing plate track for camera {track.camera_id}: "
        f"{label}  conf={pr.confidence:.2f}  hits={track.hit_count}"
    )

    if pr.plate_number:
        add_plate_cooldown(pr.plate_number)

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
        )],
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
                            if pt.camera_id == camera_id and _iou(pt.last_bbox, pr.bbox) >= PLATE_IOU_THRESH:
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
