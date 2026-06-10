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


class FaceResult:
    """Result of a face detection."""

    def __init__(self, bbox: list, embedding: np.ndarray, det_score: float):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.embedding = embedding  # 512-dim normalized vector
        self.det_score = det_score  # Detection confidence


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

            self.model = FaceAnalysis(
                name="buffalo_l",
                root="/app/data",
                providers=providers,
            )
            self.model.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace model 'buffalo_l' loaded successfully")
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

                results.append(
                    FaceResult(
                        bbox=face.bbox.tolist(),
                        embedding=embedding.astype(np.float32),
                        det_score=float(face.det_score),
                    )
                )
            return results
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []

    def extract_embedding_from_image(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from an in-memory image array.

        Args:
            img: OpenCV BGR image array

        Returns:
            512-dim embedding or None if no face detected
        """
        if self.model is None:
            return None

        faces = self.model.get(img)
        if not faces:
            logger.warning("No face detected in image")
            return None

        # Filter to keep only frontal faces for registration (slightly more lenient threshold: 30 degrees)
        frontal_faces = [f for f in faces if is_frontal_face(f, max_yaw=30.0, max_pitch=30.0, max_roll=30.0)]
        if not frontal_faces:
            logger.warning("No frontal face detected in registration image")
            return None

        # Use the largest face (by bounding box area) among the frontal faces
        largest_face = max(frontal_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        embedding = largest_face.normed_embedding
        if embedding is None:
            embedding = largest_face.embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding.astype(np.float32)


# Module-level singleton
face_engine = FaceEngine()
