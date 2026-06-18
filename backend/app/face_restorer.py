"""
Face Restorer: CodeFormer model wrapper using ONNX Runtime.

Implements face restoration to enhance low-quality face crops.
"""

import os
import logging
from typing import Optional
import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class FaceRestorer:
    """
    Singleton class managing the CodeFormer ONNX model for face restoration.
    """

    def __init__(self):
        self.session = None
        self.enabled = os.getenv("CODEFORMER_ENABLED", "true").lower() == "true"
        # Fidelity weight: 0.0 = max restoration (may hallucinate), 1.0 = preserve original
        # Default raised to 0.9 for CCTV: less hallucination, more natural result
        try:
            self.fidelity = float(os.getenv("CODEFORMER_FIDELITY", "0.9"))
        except ValueError:
            self.fidelity = 0.9
        # Minimum face dimension (pixels) required before running CodeFormer.
        # After Real-ESRGAN x4, a 64px face becomes 256px — well above this gate.
        try:
            self.min_face_size = int(os.getenv("CODEFORMER_MIN_FACE_SIZE", "128"))
        except ValueError:
            self.min_face_size = 128
            
        self._initialized = False

    def initialize(self):
        """Load the CodeFormer ONNX model."""
        if self._initialized:
            return

        if not self.enabled:
            logger.info("Face Restorer is disabled via configuration.")
            self._initialized = True
            return

        logger.info("Initializing Face Restorer (ONNX)...")

        # Determine model file path dynamically
        model_path = os.getenv("CODEFORMER_MODEL_PATH")
        if not model_path:
            # Check standard paths
            paths_to_check = [
                "/app/data/codeformer.onnx",
                "backend/data/codeformer.onnx",
                "data/codeformer.onnx",
                os.path.expanduser("~/data/codeformer.onnx")
            ]
            for p in paths_to_check:
                if os.path.exists(p):
                    model_path = p
                    break

        if not model_path or not os.path.exists(model_path):
            logger.error(f"CodeFormer ONNX model not found. Checked paths: {paths_to_check if not model_path else model_path}")
            logger.warning("Face Restorer will be disabled (failed to find model file)")
            self.enabled = False
            self._initialized = True
            return

        logger.info(f"Loading CodeFormer ONNX model from: {model_path}")

        try:
            # Set execution providers (similar to face_engine.py)
            env_provider = os.getenv("ONNX_PROVIDER")
            available_providers = ort.get_available_providers()
            if env_provider:
                providers = [env_provider, "CPUExecutionProvider"]
                logger.info(f"Using environment-specified ONNX provider: {env_provider}")
            else:
                providers = [
                    "CoreMLExecutionProvider",       # Apple Silicon (macOS)
                    "CUDAExecutionProvider",          # NVIDIA GPU
                    "ROCmExecutionProvider",          # AMD GPU
                    "OpenVINOExecutionProvider",      # Intel CPU/GPU
                    "DmlExecutionProvider",           # Windows DirectML
                    "CPUExecutionProvider",           # CPU Fallback
                ]
                logger.info(f"Attempting to load ONNX providers for CodeFormer: {providers}")

            # Filter providers to only include those available in the current runtime environment
            providers = [p for p in providers if p in available_providers]
            logger.info(f"Filtered available ONNX providers: {providers}")

            self.session = ort.InferenceSession(model_path, providers=providers)
            logger.info("CodeFormer ONNX model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load CodeFormer ONNX model: {e}")
            logger.warning("Face Restorer will be disabled")
            self.enabled = False

        self._initialized = True

    def is_enabled(self) -> bool:
        """Check if Face Restorer is enabled and initialized successfully."""
        if not self._initialized:
            self.initialize()
        return self.enabled and self.session is not None

    def restore_face(self, face_crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Restore a low-quality face crop using CodeFormer.

        Args:
            face_crop_bgr: Low-quality face image as a BGR numpy array.

        Returns:
            Restored face image as a BGR numpy array (512x512), or None if failed.
        """
        if not self.is_enabled():
            return None

        if face_crop_bgr is None or face_crop_bgr.size == 0:
            logger.warning("Empty face crop passed to FaceRestorer")
            return None

        h, w = face_crop_bgr.shape[:2]
        if h < self.min_face_size or w < self.min_face_size:
            logger.debug(
                f"Face {w}x{h} is below min size ({self.min_face_size}px) for CodeFormer — "
                "run Real-ESRGAN first to upscale"
            )
            return None

        try:
            # 1. Preprocessing: Resize to 512x512 (required by CodeFormer)
            img = cv2.resize(face_crop_bgr, (512, 512))
            
            # 2. Convert to RGB (CodeFormer expects RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 3. Normalize from [0, 255] to [-1, 1] (img - 0.5) / 0.5
            img = img.astype(np.float32) / 255.0
            img = (img - 0.5) / 0.5
            
            # 4. Transpose to [C, H, W] and add batch dimension [1, C, H, W]
            img = img.transpose((2, 0, 1))
            img = np.expand_dims(img, axis=0)

            # 5. Prepare inputs
            # 'x': Input image tensor (float32)
            # 'w': Fidelity weight tensor (double/float64, scalar shape [])
            w_val = np.array(self.fidelity, dtype=np.float64)
            inputs = {'x': img, 'w': w_val}

            # 6. Run inference (synchronized to prevent DirectML crashes)
            from app.gpu_lock import inference_lock
            with inference_lock:
                outputs = self.session.run(['y'], inputs)
            restored_tensor = outputs[0]

            # 7. Postprocessing: Remove batch dimension, transpose to [H, W, C]
            restored_img = restored_tensor[0].transpose((1, 2, 0))
            
            # 8. Denormalize from [-1, 1] to [0, 255]
            restored_img = (restored_img * 0.5 + 0.5) * 255.0
            restored_img = np.clip(restored_img, 0, 255).astype(np.uint8)
            
            # 9. Convert back from RGB to BGR
            restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
            
            return restored_img

        except Exception as e:
            logger.error(f"Error restoring face: {e}")
            return None


# Module-level singleton
face_restorer = FaceRestorer()
