"""
Face Engine: InsightFace model wrapper.

Handles face detection and embedding extraction.
All Qdrant and FastAPI dependencies have been removed as the AI Worker is stateless
and the Go Control Plane manages vector search.
"""

import os
import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Minimum Laplacian variance for a face crop to be considered "sharp enough".
# Higher values are stricter. Typical range: 30-100 depending on camera quality.
SHARPNESS_THRESHOLD = float(os.getenv("FACE_SHARPNESS_THRESHOLD", "50.0"))


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


def compute_frontality(face, max_yaw: float = 20.0, max_pitch: float = 20.0, max_roll: float = 20.0) -> float:
    """
    Compute a 0.0–1.0 frontality score from InsightFace pose angles.

    1.0 = perfectly frontal (yaw ≈ pitch ≈ roll ≈ 0).
    0.0 = face at the maximum allowed angle threshold.

    The score is the average of per-axis ratios:
        axis_score = 1.0 - clamp(|angle| / max_angle, 0, 1)

    Args:
        face: InsightFace detection object with .pose attribute.
        max_yaw:   Maximum yaw   considered frontal (degrees).
        max_pitch:  Maximum pitch considered frontal (degrees).
        max_roll:   Maximum roll  considered frontal (degrees).

    Returns:
        Frontality score (float, 0.0–1.0).  Higher = more frontal.
    """
    pose = face.get("pose")
    if pose is None:
        return 1.0  # Assume frontal if pose estimation unavailable

    pitch, yaw, roll = pose
    yaw_score = 1.0 - min(abs(yaw) / max_yaw, 1.0)
    pitch_score = 1.0 - min(abs(pitch) / max_pitch, 1.0)
    roll_score = 1.0 - min(abs(roll) / max_roll, 1.0)
    return (yaw_score + pitch_score + roll_score) / 3.0


class FaceResult:
    """Result of a face detection."""

    def __init__(self, bbox: list, embedding: np.ndarray, det_score: float, sharpness: float = 0.0, frontality: float = 1.0):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.embedding = embedding  # 512-dim normalized vector
        self.det_score = det_score  # Detection confidence
        self.sharpness = sharpness  # Laplacian variance of the face crop (higher = sharper)
        self.frontality = frontality  # 0.0–1.0 pose frontality score (higher = more frontal)


def is_frontal_face(face, max_yaw: float = 20.0, max_pitch: float = 20.0, max_roll: float = 20.0) -> bool:
    """
    Check if the face is looking straight (frontal face) based on pose angles.
    InsightFace returns pose as [pitch, yaw, roll] in degrees.
    """
    pose = face.get("pose")
    if pose is None:
        return True  # Fallback if pose estimation is not available
    
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

    def initialize(self):
        """Load the InsightFace model."""
        if self._initialized:
            return

        logger.info("Initializing Face Engine...")

        # Load InsightFace model
        try:
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
            )
            
            # NOTE: We MUST use 640x640 for CoreMLExecutionProvider. 
            # CoreML compiles the ONNX model into a static .mlmodel, and the SCRFD 
            # detector used in buffalo_l has hardcoded anchor shapes. Changing this 
            # to 320x320 causes a shape mismatch crash (Bus error 10) on macOS.
            det_size_val = int(os.getenv("FACE_DETECTION_SIZE", "640"))
            self.model.prepare(ctx_id=0, det_size=(det_size_val, det_size_val))
            logger.info(f"InsightFace model 'buffalo_l' loaded successfully with det_size={det_size_val}x{det_size_val}")
        except Exception as e:
            logger.warning(f"Failed to load InsightFace model: {e}")
            logger.warning("Face engine will operate in mock mode (no real detection)")
            self.model = None

        self._initialized = True

    def detect_faces(self, frame: np.ndarray) -> list[FaceResult]:
        """
        Detect all faces in a single frame.

        Args:
            frame: BGR image as numpy array (from OpenCV)

        Returns:
            List of FaceResult with bounding box, embedding, and detection score
        """
        if self.model is None:
            return []

        try:
            faces = self.model.get(frame)
            results = []
            for face in faces:
                if face.det_score < 0.5:  # MIN_DET_SCORE
                    continue

                # Filter out non-frontal faces for real-time tracking
                if not is_frontal_face(face, max_yaw=20.0, max_pitch=20.0, max_roll=20.0):
                    logger.debug(f"Skipped non-frontal face (pose: {face.get('pose')})")
                    continue

                # Normalize embedding for cosine similarity
                embedding = face.normed_embedding
                if embedding is None:
                    embedding = face.embedding
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm

                # Compute sharpness score for best-frame selection
                bbox_list = face.bbox.tolist()
                sharpness = compute_sharpness(frame, bbox_list)

                # Compute frontality score (how straight the face is looking)
                frontality = compute_frontality(face, max_yaw=20.0, max_pitch=20.0, max_roll=20.0)

                results.append(
                    FaceResult(
                        bbox=bbox_list,
                        embedding=embedding.astype(np.float32),
                        det_score=float(face.det_score),
                        sharpness=sharpness,
                        frontality=frontality,
                    )
                )
            return results
        except Exception as e:
            logger.error(f"Face detection error: {e}")
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

        # 2. Frontality Gate: Check yaw, pitch, roll (strict 15.0 degrees for registration)
        pose = face.get("pose")
        if pose is not None:
            pitch, yaw, roll = pose
            max_angle = 15.0
            if abs(pitch) > max_angle or abs(yaw) > max_angle or abs(roll) > max_angle:
                return None, f"Face is not looking straight (yaw: {abs(yaw):.1f}°, pitch: {abs(pitch):.1f}°, roll: {abs(roll):.1f}°). Maximum allowed is {max_angle}°."

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
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding.astype(np.float32), None



# Module-level singleton
face_engine = FaceEngine()
