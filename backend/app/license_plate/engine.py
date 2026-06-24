"""
LicensePlateEngine — wraps the YOLO plate + char detection pipeline.

Usage:
    engine = LicensePlateEngine()
    results = engine.detect(frame_bgr)  # returns list[PlateResult]
"""

from __future__ import annotations
import logging
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CharDet:
    x_norm: float      # 0.0–1.0 within plate bbox (left→right, position-based)
    char: str
    confidence: float
    # YOLO-format bbox in deskewed crop image coords (normalized 0-1)
    cx: float = 0.0   # center x
    cy: float = 0.0   # center y
    bw: float = 0.0   # width
    bh: float = 0.0   # height


@dataclass
class PlateResult:
    plate_number: Optional[str]   # normalized plate text, e.g. "กข 1234"
    confidence: float             # 0.0–1.0
    bbox: list[float]             # [x1, y1, x2, y2] in original image coords
    plate_type: str               # "standard" | "commercial" | "unknown"
    province: Optional[str]       # Thai province name if detected
    raw_text: str                 # raw chars before validation
    chars: list = field(default_factory=list)  # list[CharDet] for multi-frame assembly
    crop_bytes: Optional[bytes] = None  # JPEG of deskewed plate crop (for training)


# ── CHAR_LABEL_MAP ────────────────────────────────────────────────────────────
CHAR_LABEL_MAP: dict[str, str] = {
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
    '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
    'A01': 'ก', 'A02': 'ข', 'A04': 'ค', 'A06': 'ฆ',
    'A07': 'ง', 'A08': 'จ', 'A09': 'ฉ', 'A10': 'ช',
    'A11': 'ซ', 'A12': 'ฌ', 'A13': 'ญ', 'A14': 'ฎ',
    'A15': 'ฏ', 'A16': 'ฐ', 'A17': 'ฑ', 'A18': 'ฒ',
    'A19': 'ณ', 'A20': 'ด', 'A21': 'ต', 'A22': 'ถ',
    'A23': 'ท', 'A24': 'ธ', 'A25': 'น', 'A26': 'บ',
    'A27': 'ป', 'A28': 'ผ', 'A29': 'ฝ', 'A30': 'พ',
    'A31': 'ฟ', 'A32': 'ภ', 'A33': 'ม', 'A34': 'ย',
    'A35': 'ร', 'A36': 'ล', 'A37': 'ว', 'A38': 'ศ',
    'A39': 'ษ', 'A40': 'ส', 'A41': 'ห', 'A42': 'ฬ',
    'A43': 'อ', 'A44': 'ฮ',
    'ACR': 'อ่างทอง', 'ATG': 'อ่างทอง', 'AYA': 'อยุธยา',
    'BKK': 'กรุงเทพ', 'BKN': 'บึงกาฬ', 'BRM': 'บุรีรัมย์',
    'CBI': 'ชลบุรี', 'CCO': 'ฉะเชิงเทรา', 'CMI': 'เชียงใหม่',
    'CNT': 'ชัยนาท', 'CPM': 'ชัยภูมิ', 'CPN': 'ชุมพร',
    'CRI': 'เชียงราย', 'CTI': 'ชัยนาท',
    'KBI': 'กระบี่', 'KKN': 'ขอนแก่น', 'KPT': 'กาฬสินธุ์',
    'KRI': 'กาญจนบุรี', 'KSN': 'กาฬสินธุ์',
    'LEI': 'เลย', 'LPG': 'ลำปาง', 'LPN': 'ลำพูน', 'LRI': 'ลพบุรี',
    'MDH': 'มหาสารคาม', 'MKM': 'มุกดาหาร',
    'NAN': 'น่าน', 'NBI': 'หนองบัวลำภู', 'NBP': 'นนทบุรี',
    'NKI': 'หนองคาย', 'NMA': 'นครราชสีมา', 'NPM': 'นครปฐม',
    'NPT': 'นครพนม', 'NRT': 'นราธิวาส', 'NSN': 'นครสวรรค์',
    'NST': 'นครศรีธรรมราช', 'NWT': 'นครนายก', 'NYK': 'นนทบุรี',
    'PBI': 'ปราจีนบุรี', 'PCT': 'ประจวบคีรีขันธ์',
    'PKN': 'ประจวบฯ', 'PKT': 'ภูเก็ต',
    'PLG': 'พัทลุง', 'PLK': 'พิษณุโลก',
    'PNA': 'พังงา', 'PNB': 'เพชรบูรณ์',
    'PRE': 'แพร่', 'PRI': 'ปทุมธานี', 'PTE': 'พัทลุง', 'PTN': 'ปัตตานี',
    'PYO': 'พะเยา',
    'RBR': 'ราชบุรี', 'RET': 'ร้อยเอ็ด', 'RNG': 'ระนอง', 'RYG': 'ระยอง',
    'SBR': 'สระบุรี', 'SKA': 'สงขลา', 'SKM': 'สกลนคร',
    'SKN': 'สมุทรสาคร', 'SKW': 'สมุทรสงคราม',
    'SNI': 'สิงห์บุรี', 'SNK': 'สกลนคร',
    'SPB': 'สุพรรณบุรี', 'SPK': 'สมุทรปราการ',
    'SRI': 'สุรินทร์', 'SRN': 'สุราษฎร์ธานี',
    'SSK': 'ศรีสะเกษ', 'STI': 'สตูล', 'STN': 'สุโขทัย',
    'TAK': 'ตาก', 'TRG': 'ตรัง', 'TRT': 'ตราด',
    'UBN': 'อุบลราชธานี', 'UDN': 'อุดรธานี',
    'UTI': 'อุทัยธานี', 'UTT': 'อุตรดิตถ์',
    'YLA': 'ยะลา', 'YST': 'ยโสธร',
}

PROVINCE_CODES = {k for k in CHAR_LABEL_MAP if len(k) == 3 and not k.startswith('A')}


def _deskew(crop: np.ndarray) -> np.ndarray:
    import cv2
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=max(20, crop.shape[1] // 5),
                            maxLineGap=10)
    if lines is None:
        return crop
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) > 5:
            angle = np.degrees(np.arctan2(dy, dx))
            if abs(angle) < 45:
                angles.append(angle)
    if not angles:
        return crop
    median = float(np.median(angles))
    if abs(median) < 1.0 or abs(median) > 30.0:
        return crop
    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median, 1.0)
    return cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _pad_square(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = image.shape[:2]
    s = max(h, w)
    canvas = np.full((s, s, 3), 114, dtype=np.uint8)
    xp, yp = (s - w) // 2, (s - h) // 2
    canvas[yp:yp + h, xp:xp + w] = image
    return canvas, xp, yp


def _enhance_plate_crop(crop: np.ndarray) -> np.ndarray:
    import cv2
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced_lab = cv2.merge([clahe.apply(l_ch), a_ch, b_ch])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(enhanced, -1, kernel)


def _binarize_plate(crop: np.ndarray) -> np.ndarray:
    import cv2
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(binary == 255) / binary.size < 0.5:
        binary = cv2.bitwise_not(binary)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _resolve_models_dir() -> str:
    """
    Resolve the model directory using the same strategy as face_engine:
      1. $FACE_DATA_ROOT/models  (explicit override)
      2. /app/data/models        (Docker container)
      3. data/models             (CWD = backend/)
      4. backend/data/models     (CWD = project root)
    """
    data_root = os.getenv("FACE_DATA_ROOT")
    if not data_root:
        if os.path.isdir("/app/data"):
            data_root = "/app/data"
        elif os.path.isdir("data"):
            data_root = "data"
        elif os.path.isdir("backend/data"):
            data_root = "backend/data"
        else:
            data_root = "data"
    # Return an ABSOLUTE path. ultralytics changes the working directory during
    # training, so any relative path resolved here would break afterwards (model
    # save, ONNX export, version dir, upload). Resolving to absolute up front
    # keeps every downstream path stable regardless of CWD.
    return os.path.abspath(os.path.join(data_root, "models"))

_MODELS_DIR = _resolve_models_dir()


def _resolve_providers() -> list[str]:
    """
    Pick onnxruntime execution providers based on env var or auto-detect.

    Set ONNX_PROVIDER to override:
        DmlExecutionProvider          — Windows AMD/Intel GPU (onnxruntime-directml)
        CUDAExecutionProvider         — NVIDIA GPU (onnxruntime-gpu)
        CoreMLExecutionProvider       — Apple Silicon (onnxruntime)
        CPUExecutionProvider          — always works
    """
    override = os.environ.get('ONNX_PROVIDER', '').strip()
    if override:
        return [override, 'CPUExecutionProvider']

    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        for preferred in ('CUDAExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider'):
            if preferred in available:
                return [preferred, 'CPUExecutionProvider']
    except Exception:
        pass
    return ['CPUExecutionProvider']


class LicensePlateEngine:
    """
    Loads YOLO models once; call detect() per frame.

    Prefers ONNX models (thai_plate_yolo11n.onnx / thai_char_yolo26s.onnx)
    for cross-platform GPU support via onnxruntime.  Falls back to .pt
    (ultralytics) if ONNX files are not present.

    GPU selection:
        - NVIDIA       → CUDAExecutionProvider  (pip install onnxruntime-gpu)
        - AMD / Intel  → DmlExecutionProvider   (pip install onnxruntime-directml)
        - Apple M-chip → CoreMLExecutionProvider (pip install onnxruntime)
        - Force via    → ONNX_PROVIDER env var
    """

    def __init__(self):
        self._plate_model = None   # YoloOnnxSession or ultralytics YOLO
        self._char_model  = None
        self._use_onnx    = False
        self._char_names: dict[int, str] = {}
        self._load()

    def _load(self):
        plate_onnx = os.path.join(_MODELS_DIR, 'thai_plate_yolo11n.onnx')
        char_onnx  = os.path.join(_MODELS_DIR, 'thai_char_yolo26s.onnx')
        char_names_json = os.path.join(_MODELS_DIR, 'thai_char_yolo26s_names.json')

        if os.path.exists(plate_onnx) and os.path.exists(char_onnx):
            self._load_onnx(plate_onnx, char_onnx, char_names_json)
        else:
            self._load_pt()

    def _load_onnx(self, plate_onnx: str, char_onnx: str, names_json: str):
        try:
            import json
            from .onnx_infer import YoloOnnxSession

            providers = _resolve_providers()
            logger.info("Loading ONNX models with providers: %s", providers)

            if os.path.exists(names_json):
                with open(names_json, encoding='utf-8') as f:
                    raw = json.load(f)
                self._char_names = {int(k): v for k, v in raw.items()}
                logger.debug("Loaded %d char class names from %s", len(self._char_names), names_json)
            else:
                logger.warning("names JSON not found at %s — class names will be numeric", names_json)

            logger.info("Loading plate model: %s", os.path.basename(plate_onnx))
            self._plate_model = YoloOnnxSession(plate_onnx, providers, names={0: 'license-plate'})

            logger.info("Loading char model: %s", os.path.basename(char_onnx))
            self._char_model  = YoloOnnxSession(char_onnx, providers, names=self._char_names)

            self._use_onnx = True
            active_provider = self._plate_model.session.get_providers()[0]
            logger.info(
                "License plate engine ready (ONNX) — provider=%s  plate_imgsz=%d  char_imgsz=%d",
                active_provider,
                self._plate_model.imgsz,
                self._char_model.imgsz,
            )
        except Exception as e:
            logger.warning("ONNX load failed (%s) — falling back to .pt", e)
            self._load_pt()

    def _load_pt(self):
        try:
            import torch
            from ultralytics import YOLO

            device = 'cpu'
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'

            plate_pt = os.path.join(_MODELS_DIR, 'thai_plate_yolo11n.pt')
            char_pt  = os.path.join(_MODELS_DIR, 'thai_char_yolo26s.pt')

            if os.path.exists(plate_pt) and os.path.exists(char_pt):
                logger.info("Loading plate model (.pt): %s", plate_pt)
                self._plate_model = YOLO(plate_pt)
                logger.info("Loading char model (.pt): %s", char_pt)
                self._char_model  = YOLO(char_pt)
                self._plate_model.to(device)
                self._char_model.to(device)
                self._use_onnx = False
                logger.info("License plate engine ready (.pt) — device=%s", device)
            else:
                logger.error(
                    "No model files found in %s — plate detection disabled. "
                    "Run: python future/export_to_onnx.py",
                    _MODELS_DIR,
                )
        except Exception as e:
            logger.error("Failed to load .pt models: %s", e)

    @property
    def ready(self) -> bool:
        return self._plate_model is not None and self._char_model is not None

    # ── public API ──────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[PlateResult]:
        """Run full plate detection pipeline on a BGR frame."""
        if not self.ready:
            return []
        if self._use_onnx:
            return self._detect_onnx(frame)
        return self._detect_pt(frame)

    # ── ONNX path ───────────────────────────────────────────────────────────

    def _detect_onnx(self, frame: np.ndarray) -> list[PlateResult]:
        import cv2
        from .validate import LicensePlateValidator

        h, w = frame.shape[:2]
        results: list[PlateResult] = []
        try:
            # Stage 1 — locate all plates
            padded, xp, yp = _pad_square(frame)
            plate_dets = self._plate_model.detect(padded, conf_thresh=0.1)
            if not plate_dets:
                return []

            for det in plate_dets:
                bx1, by1, bx2, by2 = [int(v) for v in det['bbox']]
                pw = bx2 - bx1
                ph = by2 - by1
                x1 = max(0, bx1 - xp - int(pw * 0.15))
                y1 = max(0, by1 - yp - int(ph * 0.20))
                x2 = min(w, bx2 - xp + int(pw * 0.15))
                y2 = min(h, by2 - yp + int(ph * 0.20))

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Stage 2 — deskew + scale
                deskewed = _deskew(crop)
                ch_h = deskewed.shape[0]
                scale = max(2.0, 80.0 / ch_h) if ch_h < 60 else (2.0 if ch_h < 120 else 1.0)
                if scale > 1.0:
                    deskewed = cv2.resize(deskewed, None, fx=scale, fy=scale,
                                          interpolation=cv2.INTER_CUBIC)

                # Stage 3 — char detection: BW → enhanced → binarized
                gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                bw = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

                pad_bw, cxp, cyp = _pad_square(bw)
                char_dets = self._char_model.detect(pad_bw, conf_thresh=0.20)
                if not char_dets:
                    enhanced = _enhance_plate_crop(deskewed)
                    pad_enh, cxp, cyp = _pad_square(enhanced)
                    char_dets = self._char_model.detect(pad_enh, conf_thresh=0.15)
                if not char_dets:
                    binarized = _binarize_plate(deskewed)
                    pad_bin, cxp, cyp = _pad_square(binarized)
                    char_dets = self._char_model.detect(pad_bin, conf_thresh=0.15)
                if not char_dets:
                    pad_raw, cxp, cyp = _pad_square(deskewed)
                    char_dets = self._char_model.detect(pad_raw, conf_thresh=0.15)
                if not char_dets:
                    continue

                # Stage 4 — assemble chars
                plate_results = self._assemble_chars(
                    char_dets, cxp, cyp, x1, y1, x2, y2,
                    LicensePlateValidator, crop_shape=deskewed.shape[:2])
                # Attach deskewed color crop bytes for training data capture
                try:
                    ok, buf = cv2.imencode('.jpg', deskewed, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    crop_bytes = bytes(buf) if ok else None
                except Exception:
                    crop_bytes = None
                for pr in plate_results:
                    pr.crop_bytes = crop_bytes
                results.extend(plate_results)
        except Exception as e:
            logger.error("ONNX detect error: %s", e, exc_info=True)
        return results

    # ── .pt path ────────────────────────────────────────────────────────────

    def _detect_pt(self, frame: np.ndarray) -> list[PlateResult]:
        import cv2
        from .validate import LicensePlateValidator

        h, w = frame.shape[:2]
        results: list[PlateResult] = []
        try:
            padded, xp, yp = _pad_square(frame)
            yolo_out = self._plate_model(padded, imgsz=1280, conf=0.1, verbose=False)
            if not yolo_out or len(yolo_out[0].boxes) == 0:
                return []

            for box_obj in yolo_out[0].boxes:
                box = box_obj.xyxy[0].cpu().numpy().astype(int)
                pw = box[2] - box[0]
                ph = box[3] - box[1]
                x1 = max(0, box[0] - xp - int(pw * 0.15))
                y1 = max(0, box[1] - yp - int(ph * 0.20))
                x2 = min(w, box[2] - xp + int(pw * 0.15))
                y2 = min(h, box[3] - yp + int(ph * 0.20))

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                deskewed = _deskew(crop)
                ch_h = deskewed.shape[0]
                scale = max(2.0, 80.0 / ch_h) if ch_h < 60 else (2.0 if ch_h < 120 else 1.0)
                if scale > 1.0:
                    deskewed = cv2.resize(deskewed, None, fx=scale, fy=scale,
                                          interpolation=cv2.INTER_CUBIC)

                gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                bw = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

                pad_bw, cxp, cyp = _pad_square(bw)
                char_out = self._char_model(pad_bw, imgsz=640, conf=0.20, verbose=False)
                if not char_out or len(char_out[0].boxes) == 0:
                    enhanced = _enhance_plate_crop(deskewed)
                    pad_enh, cxp, cyp = _pad_square(enhanced)
                    char_out = self._char_model(pad_enh, imgsz=640, conf=0.15, verbose=False)
                if not char_out or len(char_out[0].boxes) == 0:
                    binarized = _binarize_plate(deskewed)
                    pad_bin, cxp, cyp = _pad_square(binarized)
                    char_out = self._char_model(pad_bin, imgsz=640, conf=0.15, verbose=False)
                if not char_out or len(char_out[0].boxes) == 0:
                    pad_raw, cxp, cyp = _pad_square(deskewed)
                    char_out = self._char_model(pad_raw, imgsz=640, conf=0.15, verbose=False)
                if not char_out or len(char_out[0].boxes) == 0:
                    continue

                char_dets = []
                for cb in char_out[0].boxes:
                    bx1 = cb.xyxy[0][0].item()
                    by1 = cb.xyxy[0][1].item()
                    bx2 = cb.xyxy[0][2].item()
                    by2 = cb.xyxy[0][3].item()
                    cls_name = self._char_model.names[int(cb.cls[0].item())]
                    char_dets.append({
                        'bbox': [bx1, by1, bx2, by2],
                        'confidence': float(cb.conf[0].item()),
                        'class_name': cls_name,
                    })

                plate_results = self._assemble_chars(
                    char_dets, cxp, cyp, x1, y1, x2, y2,
                    LicensePlateValidator, crop_shape=deskewed.shape[:2])
                try:
                    ok, buf = cv2.imencode('.jpg', deskewed, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    crop_bytes = bytes(buf) if ok else None
                except Exception:
                    crop_bytes = None
                for pr in plate_results:
                    pr.crop_bytes = crop_bytes
                results.extend(plate_results)
        except Exception as e:
            logger.error(".pt detect error: %s", e, exc_info=True)
        return results

    # ── shared char assembly ─────────────────────────────────────────────────

    def _assemble_chars(self, char_dets: list[dict], cxp: float, cyp: float,
                        x1: int, y1: int, x2: int, y2: int, validator,
                        crop_shape: tuple = (0, 0)) -> list[PlateResult]:
        """
        crop_shape: (h, w) of the deskewed plate crop image, used to produce
        YOLO-normalized bboxes for training data export.
        """
        chars_with_pos = []
        province_detected = None
        box_heights = []
        # raw bbox data for CharDet (in deskewed crop space)
        raw_bboxes: list[tuple] = []

        crop_h, crop_w = crop_shape if crop_shape[0] > 0 else (1, 1)

        for det in char_dets:
            bx1, by1, bx2, by2 = det['bbox']
            xc = (bx1 + bx2) / 2 - cxp
            yc = (by1 + by2) / 2 - cyp
            box_heights.append(abs(by2 - by1))
            cls_name = det['class_name']
            thai_char = CHAR_LABEL_MAP.get(cls_name, cls_name)
            conf = det['confidence']
            if cls_name in PROVINCE_CODES:
                province_detected = thai_char
                continue
            # tuple: (xc, yc, char, conf, raw_bx1, raw_by1, raw_bx2, raw_by2)
            chars_with_pos.append((xc, yc, thai_char, conf, bx1, by1, bx2, by2))

        if not chars_with_pos:
            return []

        avg_h = float(np.mean(box_heights)) if box_heights else 20.0
        sorted_chars = sorted(chars_with_pos, key=lambda c: c[0])
        rows: list[list] = []
        for ch in sorted_chars:
            best_row = None
            best_y_diff = float('inf')
            for row in rows:
                y_diff = abs(ch[1] - row[-1][1])
                if y_diff < avg_h * 0.75 and y_diff < best_y_diff:
                    best_row = row
                    best_y_diff = y_diff
            if best_row is not None:
                best_row.append(ch)
            else:
                rows.append([ch])
        rows.sort(key=lambda r: float(np.mean([c[1] for c in r])))
        plate_chars = [c for row in rows for c in row]

        raw_text = "".join(c[2] for c in plate_chars)
        avg_conf = float(np.mean([c[3] for c in plate_chars]))
        min_conf = float(np.min([c[3] for c in plate_chars]))

        n = len(plate_chars)
        chars_for_result = []
        for i, c in enumerate(plate_chars):
            # c = (xc, yc, char, conf, rbx1, rby1, rbx2, rby2)
            rbx1, rby1, rbx2, rby2 = c[4], c[5], c[6], c[7]
            # normalize raw padded-image bbox to deskewed crop space
            rcx = ((rbx1 + rbx2) / 2 - cxp) / crop_w
            rcy = ((rby1 + rby2) / 2 - cyp) / crop_h
            rbw = (rbx2 - rbx1) / crop_w
            rbh = (rby2 - rby1) / crop_h
            chars_for_result.append(CharDet(
                x_norm=(i + 0.5) / n,
                char=c[2],
                confidence=c[3],
                cx=max(0.0, min(1.0, rcx)),
                cy=max(0.0, min(1.0, rcy)),
                bw=min(1.0, max(0.0, rbw)),
                bh=min(1.0, max(0.0, rbh)),
            ))

        # Discard single-character fragments — no valid Thai plate is shorter than 3 chars.
        if len(raw_text) < 3:
            logger.debug("Discarding fragment '%s' (len=%d)", raw_text, len(raw_text))
            return []

        is_valid, normalized, _ = validator.validate(raw_text)
        if not is_valid:
            corrected = validator.correct_common_errors(raw_text)
            if corrected:
                is_valid, normalized, _ = validator.validate(corrected)

        plate_number = normalized if is_valid else None

        # Drop low-confidence invalid results early — they add noise without value.
        if plate_number is None and avg_conf < 0.30:
            logger.debug("Discarding low-conf invalid plate '%s' (avg_conf=%.2f)", raw_text, avg_conf)
            return []

        return [PlateResult(
            plate_number=plate_number,
            confidence=min_conf if plate_number else avg_conf * 0.5,
            bbox=[float(x1), float(y1), float(x2), float(y2)],
            plate_type='standard',
            province=province_detected,
            raw_text=raw_text,
            chars=chars_for_result,
        )]


# Singleton — loaded once when the worker process starts
license_plate_engine = LicensePlateEngine()
