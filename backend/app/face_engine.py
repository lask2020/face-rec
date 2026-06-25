"""
Face Engine: InsightFace model wrapper.

Handles face detection and embedding extraction.
All Qdrant and FastAPI dependencies have been removed as the AI Worker is stateless
and the Go Control Plane manages vector search.
"""

import os
import logging
import threading
import traceback
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Minimum Laplacian variance for a face crop to be considered "sharp enough".
# Higher values are stricter. Typical range: 30-100 depending on camera quality.
SHARPNESS_THRESHOLD = float(os.getenv("FACE_SHARPNESS_THRESHOLD", "50.0"))

# Minimum face bounding box size (width and height in pixels) to be tracked/processed.
MIN_FACE_SIZE = float(os.getenv("MIN_FACE_SIZE", "45.0"))


def compute_sharpness(image: np.ndarray, bbox: list) -> float:
    """
    Compute a sharpness score for a face crop using Laplacian variance.

    The Laplacian highlights edges; a blurry image has very few edges,
    producing a low variance.  A sharp image produces a high variance.

    Args:
        image: Full BGR frame (numpy array).
        bbox: [x1, y1, x2, y2] bounding box of the face.

    Returns:
        Laplacian variance (float).  Higher = sharper.
    """
    if image is None:
        return 0.0
    h, w = image.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def _estimate_pose_from_kps(kps) -> tuple[float, float, float]:
    """
    Estimate rough pitch, yaw, roll from 5 facial landmarks (kps).
    kps: numpy array of shape (5, 2)
        0: left eye
        1: right eye
        2: nose
        3: left mouth corner
        4: right mouth corner
    Returns: (pitch, yaw, roll) in degrees
    """
    if kps is None or len(kps) < 5:
        return 0.0, 0.0, 0.0

    # Extract points
    leye = kps[0]
    reye = kps[1]
    nose = kps[2]
    lmouth = kps[3]
    rmouth = kps[4]

    # Yaw: horizontal ratio of nose to eyes
    eye_center_x = (leye[0] + reye[0]) / 2.0
    eye_dist = max(1.0, reye[0] - leye[0])
    yaw_ratio = (nose[0] - eye_center_x) / eye_dist
    yaw = yaw_ratio * 90.0  # approximate scaling to degrees

    # Pitch: vertical ratio of nose to eye-mouth distance
    mouth_center_y = (lmouth[1] + rmouth[1]) / 2.0
    eye_center_y = (leye[1] + reye[1]) / 2.0
    face_height = max(1.0, mouth_center_y - eye_center_y)
    nose_ratio = (nose[1] - eye_center_y) / face_height
    # nose is typically ~45% down from eyes to mouth.
    pitch = (nose_ratio - 0.45) * 100.0  # approximate scaling

    # Roll: angle between eyes
    dy = reye[1] - leye[1]
    dx = reye[0] - leye[0]
    roll = np.degrees(np.arctan2(dy, dx))

    return pitch, yaw, roll


def compute_frontality(face, max_yaw: float = 20.0, max_pitch: float = 20.0, max_roll: float = 20.0) -> float:
    """
    Compute a score from 0 to 1 indicating how frontal a face is.
    """
    pose = face.get("pose")
    if pose is None:
        pose = _estimate_pose_from_kps(getattr(face, 'kps', None))

    pitch, yaw, roll = pose
    yaw_score = max(0.0, 1.0 - abs(yaw) / max_yaw)
    pitch_score = max(0.0, 1.0 - abs(pitch) / max_pitch)
    roll_score = max(0.0, 1.0 - abs(roll) / max_roll)
    return (yaw_score + pitch_score + roll_score) / 3.0


class FaceResult:
    """Result of a face detection."""

    def __init__(self, bbox: list, embedding: np.ndarray, det_score: float, sharpness: float = 0.0, frontality: float = 1.0, kps: Optional[list] = None):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.embedding = embedding  # 512-dim normalized vector
        self.det_score = det_score  # Detection confidence
        self.sharpness = sharpness  # Laplacian variance of the face crop (higher = sharper)
        self.frontality = frontality  # 0.0–1.0 pose frontality score (higher = more frontal)
        self.kps = kps  # [5, 2] landmarks array


def is_frontal_face(face, max_yaw: float = 20.0, max_pitch: float = 20.0, max_roll: float = 20.0) -> bool:
    """
    Check if the face is looking straight (frontal face) based on pose angles.
    """
    pose = face.get("pose")
    if pose is None:
        pose = _estimate_pose_from_kps(getattr(face, 'kps', None))
    
    pitch, yaw, roll = pose
    return abs(pitch) <= max_pitch and abs(yaw) <= max_yaw and abs(roll) <= max_roll


class FaceEngine:
    """
    Singleton managing InsightFace model.

    Responsibilities:
    - Load InsightFace model (buffalo_l) on init
    - Detect faces in frames and extract 512-dim embeddings
    """

    def __init__(self):
        self.model = None
        self.embedding_dim = 512
        self._initialized = False
        from app.gpu_lock import inference_lock
        self._inference_lock = inference_lock

    def initialize(self):
        """Load the InsightFace model."""
        if self._initialized:
            return

        logger.info("Initializing Face Engine...")

        # Load InsightFace model
        try:
            # Patch ml_dtypes for onnx on Python 3.13 to prevent missing attribute errors
            try:
                import numpy as np
                import ml_dtypes
                # List of newer data types onnx expects but older ml_dtypes lacks
                missing_types = [
                    'float4_e2m1fn', 'float8_e8m0fnu', 'float8_e4m3fn', 
                    'float8_e4m3fnuz', 'float8_e5m2', 'float8_e5m2fnuz',
                    'int4', 'uint4'
                ]
                for attr in missing_types:
                    if not hasattr(ml_dtypes, attr):
                        setattr(ml_dtypes, attr, np.float32)
            except ImportError:
                pass

            import insightface
            from insightface.app import FaceAnalysis

            # Get execution provider from environment or use fallback list
            env_provider = os.getenv("ONNX_PROVIDER")
            if env_provider:
                providers = [env_provider, "CPUExecutionProvider"]
                logger.info(f"Using environment-specified ONNX provider: {env_provider}")
            else:
                providers = [
                    "CoreMLExecutionProvider",       # Apple Silicon (macOS)
                    "CUDAExecutionProvider",          # NVIDIA GPU
                    "ROCmExecutionProvider",          # AMD GPU (Linux ROCm)
                    "MIGraphXExecutionProvider",      # AMD GPU (Linux MIGraphX)
                    "OpenVINOExecutionProvider",      # Intel CPU/GPU (OpenVINO)
                    "DmlExecutionProvider",           # Windows DirectML (AMD/Intel/NVIDIA)
                    "CPUExecutionProvider",           # CPU Fallback
                ]
                logger.info(f"Attempting to load ONNX providers: {providers}")

            # Determine models root directory dynamically for both Docker and native host environments
            data_root = os.getenv("FACE_DATA_ROOT")
            if not data_root:
                if os.path.exists("/app/data"):
                    data_root = "/app/data"
                elif os.path.exists("data"):
                    data_root = "data"
                elif os.path.exists("backend/data"):
                    data_root = "backend/data"
                else:
                    data_root = os.path.expanduser("~/.insightface")
            logger.info(f"Using face data root directory: {data_root}")

            self.model = FaceAnalysis(
                name="buffalo_l",
                root=data_root,
                providers=providers,
                allowed_modules=['detection', 'recognition']
            )
            
            # NOTE: We MUST use 640x640 for CoreMLExecutionProvider. 
            # CoreML compiles the ONNX model into a static .mlmodel, and the SCRFD 
            # detector used in buffalo_l has hardcoded anchor shapes. Changing this 
            # to 320x320 causes a shape mismatch crash (Bus error 10) on macOS.
            det_size_val = int(os.getenv("FACE_DETECTION_SIZE", "640"))
            self.model.prepare(ctx_id=0, det_size=(det_size_val, det_size_val))
            
            # FIX: Reorder SCRFD detection model outputs for DirectML compatibility.
            self._fix_det_output_order()
            
            # FIX: Override FaceAnalysis.get() to handle missing keypoints (kps=None).
            # DirectML may fail to produce valid keypoints, and the recognition model
            # (ArcFaceONNX) crashes on face_align.norm_crop(landmark=None) which calls
            # lmk.shape on None. We also add a lock for thread-safety with GPU providers.
            self._patch_face_analysis_get()
                
            logger.info(f"InsightFace model 'buffalo_l' loaded successfully with det_size={det_size_val}x{det_size_val}")
        except Exception as e:
            logger.warning(f"Failed to load InsightFace model: {e}")
            logger.warning("Face engine will operate in mock mode (no real detection)")
            self.model = None

        self._initialized = True

    def _fix_det_output_order(self):
        """
        Fix SCRFD detection model output ordering for non-CPU providers.

        DirectML (and potentially other providers) may return output tensors in a
        different order than CPUExecutionProvider. InsightFace's RetinaFace.forward()
        accesses outputs by index, assuming the order:
            [score_s8, score_s16, score_s32, bbox_s8, bbox_s16, bbox_s32, kps_s8, kps_s16, kps_s32]

        We fix this by inspecting output shapes:
            - last_dim == 1  → score tensors
            - last_dim == 4  → bbox tensors
            - last_dim == 10 → keypoint tensors
        Within each group, sort by first spatial dimension descending (larger = smaller stride = first).
        """
        det_model = self.model.models.get('detection') if self.model else None
        if det_model is None:
            return

        if not (hasattr(det_model, 'use_kps') and det_model.use_kps):
            return  # No keypoints to fix

        outputs = det_model.session.get_outputs()
        fmc = getattr(det_model, 'fmc', 3)

        if len(outputs) != fmc * 3:
            return  # Not the expected SCRFD-with-kps layout

        # Categorize outputs by their last dimension
        DIM_TO_GROUP = {1: 'score', 4: 'bbox', 10: 'kps'}
        groups = {'score': [], 'bbox': [], 'kps': []}

        for o in outputs:
            last_dim = o.shape[-1]
            if not isinstance(last_dim, int):
                logger.debug("Cannot fix output order: dynamic last dimension, skipping")
                return
            group = DIM_TO_GROUP.get(last_dim)
            if group is None:
                logger.debug(f"Cannot fix output order: unexpected last dim {last_dim} for output {o.name}")
                return
            groups[group].append(o)

        for group_name, group_list in groups.items():
            if len(group_list) != fmc:
                logger.debug(f"Cannot fix output order: expected {fmc} {group_name} outputs, got {len(group_list)}")
                return

        # Sort each group by spatial dimension (dim[0] or dim[1]) descending
        # Larger spatial dim → smaller stride → should come first in the index order
        def spatial_sort_key(o):
            # Use the first non-batch dimension that is an integer
            for d in o.shape[:-1]:
                if isinstance(d, int):
                    return -d
            return 0

        for group_list in groups.values():
            group_list.sort(key=spatial_sort_key)

        correct_names = [o.name for o in (groups['score'] + groups['bbox'] + groups['kps'])]
        if correct_names != det_model.output_names:
            logger.info(f"Fixed SCRFD output order for provider compatibility: {det_model.output_names} → {correct_names}")
            det_model.output_names = correct_names
        else:
            logger.debug("SCRFD detection output order is already correct")

    def _patch_face_analysis_get(self):
        """
        Override FaceAnalysis.get() for GPU provider compatibility.

        Problems solved:
        1. Thread safety: GPU providers (DirectML, CUDA) may not be thread-safe
           when multiple workers call session.run() concurrently → serialize with lock.
        2. Missing keypoints: DirectML may fail to produce valid kps, causing
           ArcFaceONNX.get() to crash in face_align.norm_crop(landmark=None).
           We skip recognition when kps is None instead of crashing.
        3. Resilience: Wrap each sub-model call in try/except so a single model
           failure doesn't lose the entire detection.
        """
        if self.model is None:
            return

        fa = self.model  # FaceAnalysis instance
        inference_lock = self._inference_lock

        def safe_get(img, max_num=0):
            with inference_lock:
                bboxes, kpss = fa.det_model.detect(img, max_num=max_num, metric='default')

            if bboxes.shape[0] == 0:
                return []

            ret = []
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i, 0:4]
                det_score = bboxes[i, 4]
                kps = None
                if kpss is not None:
                    kps = kpss[i]

                from insightface.app.common import Face
                face = Face(bbox=bbox, kps=kps, det_score=det_score)

                for taskname, model in fa.models.items():
                    if taskname == 'detection':
                        continue
                    # Skip recognition if keypoints are missing (prevents norm_crop crash)
                    if kps is None and taskname == 'recognition':
                        face.embedding = None
                        face.normed_embedding = None
                        continue
                    try:
                        with inference_lock:
                            model.get(img, face)
                    except Exception as e:
                        logger.warning(f"Sub-model '{taskname}' failed for face {i}: {e}")

                ret.append(face)
            return ret

        fa.get = safe_get
        logger.info("Patched FaceAnalysis.get() for GPU thread-safety and kps=None handling")

    def detect_faces(self, frame: np.ndarray) -> list[FaceResult]:
        """
        Detect all faces in a single frame.

        Args:
            frame: BGR image as numpy array (from OpenCV)

        Returns:
            List of FaceResult with bounding box, embedding, and detection score
        """
        if self.model is None or frame is None:
            return []

        try:
            faces = self.model.get(frame)
            results = []
            for face in faces:
                if face.det_score < 0.5:  # MIN_DET_SCORE
                    continue

                # Filter out extremely small faces (unrecognizable or far away)
                x1, y1, x2, y2 = face.bbox
                width = x2 - x1
                height = y2 - y1
                if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
                    logger.debug(f"Skipped small face ({width:.1f}x{height:.1f}px, min required: {MIN_FACE_SIZE}px)")
                    continue

                # Filter out non-frontal faces for real-time tracking
                if not is_frontal_face(face, max_yaw=20.0, max_pitch=20.0, max_roll=20.0):
                    logger.debug(f"Skipped non-frontal face (pose: {face.get('pose')})")
                    continue

                # Normalize embedding for cosine similarity
                embedding = face.normed_embedding
                if embedding is None:
                    embedding = face.embedding
                    if embedding is not None:
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm

                if embedding is None:
                    logger.debug("Skipped face: no embedding could be extracted (missing landmarks)")
                    continue

                # Compute sharpness score for best-frame selection
                bbox_list = face.bbox.tolist()
                sharpness = compute_sharpness(frame, bbox_list)

                # Compute frontality score (how straight the face is looking)
                frontality = compute_frontality(face, max_yaw=20.0, max_pitch=20.0, max_roll=20.0)

                kps_list = face.kps.tolist() if face.kps is not None else None

                results.append(
                    FaceResult(
                        bbox=bbox_list,
                        embedding=embedding.astype(np.float32),
                        det_score=float(face.det_score),
                        sharpness=sharpness,
                        frontality=frontality,
                        kps=kps_list,
                    )
                )
            return results
        except Exception as e:
            logger.error(f"Face detection error: {e}\n{traceback.format_exc()}")
            return []

    def extract_embedding_from_image(self, img: np.ndarray) -> tuple[Optional[np.ndarray], Optional[str]]:
        """
        Extract face embedding from an in-memory image array with a strict quality gate.

        Args:
            img: OpenCV BGR image array

        Returns:
            Tuple of (512-dim embedding or None, error_message or None)
        """
        if self.model is None:
            return None, "Face model not initialized"

        faces = self.model.get(img)
        if not faces:
            return None, "No face detected in registration image"

        # 1. Multiple Faces Gate
        if len(faces) > 1:
            return None, f"Multiple faces detected ({len(faces)} faces). Please upload a photo with only one person."

        face = faces[0]

        # 2. Frontality Gate: Check yaw + roll only (strict 15.0 degrees for registration).
        # pose model is not loaded (removed for AMD GPU/Windows compatibility), so we
        # always fall back to landmark-based estimation via _estimate_pose_from_kps.
        # NOTE: pitch is intentionally NOT gated. The landmark-based pitch estimate assumes
        # every person's nose tip sits ~45% of the way from the eye-line to the mouth-line
        # (pitch = (nose_ratio - 0.45) * 100), but that ratio varies ±0.15 across individual
        # face proportions = ±15°, which is the entire tolerance. So pitch here is dominated
        # by per-person facial geometry rather than actual head tilt and falsely rejects
        # straight-on faces (e.g. a longer-nosed subject reads as pitch≈27° while looking
        # forward). yaw (nose horizontal offset) and roll (eye-line angle) are geometrically
        # robust, so we gate on those alone.
        pose = face.get("pose")
        if pose is None:
            pose = _estimate_pose_from_kps(getattr(face, 'kps', None))
        pitch, yaw, roll = pose
        max_angle = 15.0
        if abs(yaw) > max_angle or abs(roll) > max_angle:
            return None, f"Face is not looking straight (yaw: {abs(yaw):.1f}°, roll: {abs(roll):.1f}°). Maximum allowed is {max_angle}°."

        # 3. Sharpness Gate: Check Laplacian variance (strict 60.0 threshold for registration)
        bbox_list = face.bbox.tolist()
        sharpness = compute_sharpness(img, bbox_list)
        min_sharpness = 60.0
        if sharpness < min_sharpness:
            return None, f"Face image is too blurry (sharpness score: {sharpness:.1f}, minimum required: {min_sharpness:.1f}). Please upload a clearer photo."

        # Normalize embedding for similarity search
        embedding = face.normed_embedding
        if embedding is None:
            embedding = face.embedding
            if embedding is not None:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

        if embedding is None:
            return None, "Failed to extract face embedding (landmarks missing). Try a different photo."

        return embedding.astype(np.float32), None



# Module-level singleton
face_engine = FaceEngine()
