"""
Face Super Resolver: Real-ESRGAN ONNX wrapper for CCTV face upscaling.

Runs before CodeFormer to upscale and denoise low-quality face crops.
Supports DirectML (AMD/Intel Windows), CUDA, CoreML, and CPU fallback.

The ONNX model (~67MB) is downloaded automatically on first use.
"""

import os
import logging
import math
import urllib.request
import tempfile
import shutil
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
from app.gpu_lock import inference_lock

logger = logging.getLogger(__name__)

# Set ESRGAN_MODEL_URL env var to use a custom download URL.
_MODEL_URL = os.getenv(
    "ESRGAN_MODEL_URL",
    "https://huggingface.co/onnx-community/RealESRGAN_x4plus/resolve/main/model.onnx",
)
_MODEL_FILENAME = "Real-ESRGAN-x4plus.onnx"

# Maximum tile dimension to avoid GPU OOM on large faces.
DEFAULT_TILE_SIZE = int(os.getenv("ESRGAN_TILE_SIZE", "256"))
DEFAULT_TILE_PAD = int(os.getenv("ESRGAN_TILE_PAD", "10"))


def _default_model_dir() -> str:
    """Return the first writable candidate directory for storing the model."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"),
        "/app/data",
        os.path.expanduser("~/data"),
    ]
    for d in candidates:
        d = os.path.normpath(d)
        try:
            os.makedirs(d, exist_ok=True)
            # Quick write-permission check
            test = os.path.join(d, ".write_test")
            with open(test, "w") as f:
                f.write("")
            os.remove(test)
            return d
        except OSError:
            continue
    return tempfile.gettempdir()


def _download_model(dest_path: str) -> bool:
    """
    Download Real-ESRGAN ONNX with progress logging.

    HuggingFace CDN requires a proper User-Agent — urllib's default gets 401.
    Optional: set HF_TOKEN env var for gated/private models.
    Optional: set ESRGAN_MODEL_URL env var to override the download URL.
    """
    url = _MODEL_URL
    logger.info(f"Downloading Real-ESRGAN model from {url}")
    logger.info(f"  → saving to {dest_path}")

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; face-rec/1.0; +https://github.com/)",
    }
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
        logger.info("  Using HF_TOKEN for authentication.")

    tmp_path = dest_path + ".tmp"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            last_logged_pct = -1
            chunk_size = 65536  # 64KB

            with open(tmp_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded * 100 // total_size
                        milestone = (pct // 10) * 10
                        if milestone > last_logged_pct:
                            last_logged_pct = milestone
                            logger.info(f"  Downloading Real-ESRGAN... {milestone}%")

        shutil.move(tmp_path, dest_path)
        logger.info("Real-ESRGAN model downloaded successfully.")
        return True
    except urllib.error.HTTPError as e:
        logger.error(
            f"HTTP {e.code} downloading Real-ESRGAN model from {url}\n"
            f"  If 401: set HF_TOKEN env var, or set ESRGAN_MODEL_URL to a direct download link.\n"
            f"  You can also download manually and place the file at: {dest_path}"
        )
    except Exception as e:
        logger.error(f"Failed to download Real-ESRGAN model: {e}")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return False


def _find_or_download_model() -> Optional[str]:
    """Return path to the ONNX model, downloading if necessary."""
    # 1. Explicit env override
    env_path = os.getenv("ESRGAN_MODEL_PATH")
    if env_path:
        if os.path.exists(env_path):
            return env_path
        logger.warning(f"ESRGAN_MODEL_PATH set but file not found: {env_path}")

    # 2. Search standard locations
    search_dirs = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"),
        "/app/data",
        os.path.expanduser("~/data"),
    ]
    for d in search_dirs:
        p = os.path.normpath(os.path.join(d, _MODEL_FILENAME))
        if os.path.exists(p):
            logger.info(f"Found Real-ESRGAN model at {p}")
            return p

    # 3. Auto-download into the first writable directory
    model_dir = _default_model_dir()
    dest = os.path.join(model_dir, _MODEL_FILENAME)
    if _download_model(dest):
        return dest
    return None


class FaceSuperResolver:
    """
    Singleton wrapper for Real-ESRGAN ONNX model (x4 upscale).

    Only runs when the face crop is below MIN_FACE_SIZE_FOR_ESRGAN pixels on
    its smallest side, otherwise it is a no-op (face is already large enough).
    """

    MIN_FACE_SIZE_FOR_ESRGAN = int(os.getenv("ESRGAN_MIN_FACE_SIZE", "80"))

    def __init__(self):
        self.session = None
        self.enabled = os.getenv("ESRGAN_ENABLED", "true").lower() == "true"
        self._initialized = False
        self._scale = 4

    def initialize(self):
        if self._initialized:
            return

        if not self.enabled:
            logger.info("Face Super Resolver (ESRGAN) is disabled via configuration.")
            self._initialized = True
            return

        logger.info("Initializing Face Super Resolver (Real-ESRGAN ONNX)...")

        model_path = _find_or_download_model()

        if not model_path:
            logger.error("Real-ESRGAN model unavailable (download failed). Face Super Resolver disabled.")
            self.enabled = False
            self._initialized = True
            return

        logger.info(f"Loading Real-ESRGAN ONNX model from: {model_path}")
        try:
            env_provider = os.getenv("ONNX_PROVIDER")
            available_providers = ort.get_available_providers()
            if env_provider:
                providers = [env_provider, "CPUExecutionProvider"]
            else:
                providers = [
                    "CoreMLExecutionProvider",
                    "CUDAExecutionProvider",
                    "DmlExecutionProvider",
                    "CPUExecutionProvider",
                ]
            providers = [p for p in providers if p in available_providers]
            logger.info(f"ESRGAN ONNX providers: {providers}")

            self.session = ort.InferenceSession(model_path, providers=providers)
            # Inspect model I/O to get input name and scale factor
            self._input_name = self.session.get_inputs()[0].name
            logger.info(
                f"Real-ESRGAN loaded. Input name: '{self._input_name}', scale: {self._scale}x"
            )
        except Exception as e:
            logger.error(f"Failed to load Real-ESRGAN ONNX: {e}")
            self.enabled = False

        self._initialized = True

    def is_enabled(self) -> bool:
        if not self._initialized:
            self.initialize()
        return self.enabled and self.session is not None

    def upscale(self, face_crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Upscale a face crop 4x using Real-ESRGAN.

        Skips upscaling if the face is already large (>= MIN_FACE_SIZE_FOR_ESRGAN
        on both sides) to avoid unnecessary computation on good-quality frames.

        Args:
            face_crop_bgr: BGR face image (any size).

        Returns:
            Upscaled BGR image, or the original if skipped/failed.
        """
        if not self.is_enabled():
            return face_crop_bgr

        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return face_crop_bgr

        h, w = face_crop_bgr.shape[:2]
        if h >= self.MIN_FACE_SIZE_FOR_ESRGAN and w >= self.MIN_FACE_SIZE_FOR_ESRGAN:
            logger.debug(f"Face {w}x{h} is large enough, skipping ESRGAN")
            return face_crop_bgr

        try:
            result = self._run_tiled(face_crop_bgr)
            logger.debug(f"ESRGAN upscaled {w}x{h} → {result.shape[1]}x{result.shape[0]}")
            return result
        except Exception as e:
            logger.error(f"ESRGAN upscale failed: {e}")
            return face_crop_bgr

    def _run_tiled(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Run Real-ESRGAN with tiled inference to manage GPU memory.
        Tiles overlap by TILE_PAD pixels to avoid seam artifacts.
        """
        tile_size = DEFAULT_TILE_SIZE
        tile_pad = DEFAULT_TILE_PAD
        scale = self._scale

        h, w = img_bgr.shape[:2]

        # BGR → RGB, [0,255] → [0,1], HWC → CHW
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_chw = img_rgb.transpose(2, 0, 1)  # (C, H, W)

        output = np.zeros((3, h * scale, w * scale), dtype=np.float32)

        tiles_x = math.ceil(w / tile_size)
        tiles_y = math.ceil(h / tile_size)

        for iy in range(tiles_y):
            for ix in range(tiles_x):
                # Source tile in input image (with overlap padding)
                src_x1 = max(ix * tile_size - tile_pad, 0)
                src_y1 = max(iy * tile_size - tile_pad, 0)
                src_x2 = min((ix + 1) * tile_size + tile_pad, w)
                src_y2 = min((iy + 1) * tile_size + tile_pad, h)

                tile_in = img_chw[:, src_y1:src_y2, src_x1:src_x2][np.newaxis]  # (1,C,H,W)

                with inference_lock:
                    tile_out = self.session.run(None, {self._input_name: tile_in})[0][0]  # (C,H*s,W*s)

                # How many pixels of padding exist on each side of this tile's output
                pad_left = (ix * tile_size - src_x1) * scale
                pad_top = (iy * tile_size - src_y1) * scale

                # Destination in the output canvas
                dst_x1 = ix * tile_size * scale
                dst_y1 = iy * tile_size * scale
                dst_x2 = min((ix + 1) * tile_size * scale, w * scale)
                dst_y2 = min((iy + 1) * tile_size * scale, h * scale)

                copy_w = dst_x2 - dst_x1
                copy_h = dst_y2 - dst_y1

                output[:, dst_y1:dst_y2, dst_x1:dst_x2] = tile_out[
                    :, pad_top : pad_top + copy_h, pad_left : pad_left + copy_w
                ]

        # CHW → HWC, [0,1] → [0,255], RGB → BGR
        output_bgr = (np.clip(output, 0, 1).transpose(1, 2, 0) * 255.0).astype(np.uint8)
        return cv2.cvtColor(output_bgr, cv2.COLOR_RGB2BGR)


face_super_resolver = FaceSuperResolver()
