"""
LicensePlateEngine — wraps the YOLO plate + char detection pipeline.

Usage:
    engine = LicensePlateEngine()
    results = engine.detect(frame_bgr)  # returns list[PlateResult]
"""

from __future__ import annotations
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlateResult:
    plate_number: Optional[str]   # normalized plate text, e.g. "กข 1234"
    confidence: float             # 0.0–1.0
    bbox: list[float]             # [x1, y1, x2, y2] in original image coords
    plate_type: str               # "standard" | "commercial" | "unknown"
    province: Optional[str]       # Thai province name if detected
    raw_text: str                 # raw chars before validation


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


class LicensePlateEngine:
    """Loads YOLO models once; call detect() per frame."""

    # Model paths relative to wherever the worker is run from
    _PLATE_MODEL = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'models', 'thai_plate_yolo11n.pt')
    _CHAR_MODEL  = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'models', 'thai_char_yolo26s.pt')

    def __init__(self):
        self._plate_model = None
        self._char_model = None
        self._device = 'cpu'
        self._load()

    def _load(self):
        try:
            import torch
            from ultralytics import YOLO

            if torch.cuda.is_available():
                self._device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = 'mps'

            plate_path = os.path.abspath(self._PLATE_MODEL)
            char_path  = os.path.abspath(self._CHAR_MODEL)

            if os.path.exists(plate_path) and os.path.exists(char_path):
                self._plate_model = YOLO(plate_path)
                self._char_model  = YOLO(char_path)
                self._plate_model.to(self._device)
                self._char_model.to(self._device)
                print(f"[LicensePlateEngine] loaded on {self._device}")
            else:
                print(f"[LicensePlateEngine] model files not found — plate detection disabled")
        except Exception as e:
            print(f"[LicensePlateEngine] load error: {e}")

    @property
    def ready(self) -> bool:
        return self._plate_model is not None and self._char_model is not None

    def detect(self, frame: np.ndarray) -> list[PlateResult]:
        """Run full plate detection pipeline on a BGR frame."""
        if not self.ready:
            return []
        import cv2
        from .validate import LicensePlateValidator

        h, w = frame.shape[:2]
        results: list[PlateResult] = []

        try:
            # Stage 1 — locate plate
            padded, xp, yp = _pad_square(frame)
            yolo_out = self._plate_model(padded, imgsz=1280, conf=0.1, verbose=False)
            if not yolo_out or len(yolo_out[0].boxes) == 0:
                return []

            # Pick largest box
            boxes = sorted(yolo_out[0].boxes,
                           key=lambda b: ((b.xyxy[0][2] - b.xyxy[0][0]) *
                                          (b.xyxy[0][3] - b.xyxy[0][1])).item(),
                           reverse=True)
            box = boxes[0].xyxy[0].cpu().numpy().astype(int)
            x1 = max(0, box[0] - xp - int((box[2] - box[0]) * 0.15))
            y1 = max(0, box[1] - yp - int((box[3] - box[1]) * 0.20))
            x2 = min(w, box[2] - xp + int((box[2] - box[0]) * 0.15))
            y2 = min(h, box[3] - yp + int((box[3] - box[1]) * 0.20))

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return []

            # Stage 2 — deskew + scale
            deskewed = _deskew(crop)
            ch, cw = deskewed.shape[:2]
            scale = max(2.0, 80.0 / ch) if ch < 60 else (2.0 if ch < 120 else 1.0)
            if scale > 1.0:
                deskewed = cv2.resize(deskewed, None, fx=scale, fy=scale,
                                      interpolation=cv2.INTER_CUBIC)

            # Stage 3 — YOLO char detection (BW+contrast first, fallback raw)
            gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            bw = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

            pad_bw, cxp, cyp = _pad_square(bw)
            char_out = self._char_model(pad_bw, imgsz=640, conf=0.20, verbose=False)
            if not char_out or len(char_out[0].boxes) == 0:
                pad_raw, cxp, cyp = _pad_square(deskewed)
                char_out = self._char_model(pad_raw, imgsz=640, conf=0.15, verbose=False)

            if not char_out or len(char_out[0].boxes) == 0:
                return []

            # Stage 4 — assemble chars left-to-right by row
            chars_with_pos = []
            province_detected = None
            for cb in char_out[0].boxes:
                bx1 = cb.xyxy[0][0].item() - cxp
                by1 = cb.xyxy[0][1].item() - cyp
                bx2 = cb.xyxy[0][2].item() - cxp
                by2 = cb.xyxy[0][3].item() - cyp
                xc, yc = (bx1 + bx2) / 2, (by1 + by2) / 2
                cls_name = self._char_model.names[int(cb.cls[0].item())]
                thai_char = CHAR_LABEL_MAP.get(cls_name, cls_name)
                conf = float(cb.conf[0].item())
                if cls_name in PROVINCE_CODES:
                    province_detected = thai_char
                    continue
                chars_with_pos.append((xc, yc, thai_char, conf))

            if not chars_with_pos:
                return []

            avg_h = np.mean([abs(cb.xyxy[0][3].item() - cb.xyxy[0][1].item())
                             for cb in char_out[0].boxes])
            sorted_chars = sorted(chars_with_pos, key=lambda c: c[0])
            rows: list[list] = []
            for ch in sorted_chars:
                placed = False
                for row in rows:
                    if abs(ch[1] - row[-1][1]) < avg_h * 0.75:
                        row.append(ch)
                        placed = True
                        break
                if not placed:
                    rows.append([ch])
            rows.sort(key=lambda r: np.mean([c[1] for c in r]))
            plate_chars = [c for row in rows for c in row]

            raw_text = "".join(c[2] for c in plate_chars)
            avg_conf = float(np.mean([c[3] for c in plate_chars]))
            min_conf = float(np.min([c[3] for c in plate_chars]))

            # Stage 5 — validate / normalize
            is_valid, normalized, _ = LicensePlateValidator.validate(raw_text)
            if not is_valid:
                corrected = LicensePlateValidator.correct_common_errors(raw_text)
                if corrected:
                    is_valid, normalized, _ = LicensePlateValidator.validate(corrected)

            plate_number = normalized if is_valid else None
            plate_type = "standard"

            results.append(PlateResult(
                plate_number=plate_number,
                confidence=min_conf if plate_number else avg_conf * 0.5,
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                plate_type=plate_type,
                province=province_detected,
                raw_text=raw_text,
            ))

        except Exception as e:
            print(f"[LicensePlateEngine] detect error: {e}")

        return results


# Singleton — loaded once when the worker process starts
license_plate_engine = LicensePlateEngine()
