"""
ONNX inference helpers for YOLO models.

Handles two output formats produced by ultralytics export:
  - Pre-NMS  : (1, 4+nc, anchors)   — typical for detection models
  - Post-NMS : (1, max_det, 6)      — [x1,y1,x2,y2,conf,cls]

Execution providers are selected at construction time, so the same
class works for CPU / CUDA / DirectML / CoreML.
"""

from __future__ import annotations
import logging
import os
import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── letterbox / scale helpers ─────────────────────────────────────────────────

def _letterbox(img: np.ndarray, new_shape: tuple[int, int]) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize with grey padding to maintain aspect ratio."""
    h, w = img.shape[:2]
    ratio = min(new_shape[0] / h, new_shape[1] / w)
    new_w = int(round(w * ratio))
    new_h = int(round(h * ratio))
    img_r = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (new_shape[1] - new_w) / 2
    pad_h = (new_shape[0] - new_h) / 2
    top    = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left   = int(round(pad_w - 0.1))
    right  = int(round(pad_w + 0.1))
    img_p = cv2.copyMakeBorder(img_r, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img_p, ratio, (pad_w, pad_h)


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """cx, cy, w, h  →  x1, y1, x2, y2  (in-place safe)."""
    out = np.empty_like(boxes)
    out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return out


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou < iou_thresh]
    return keep


def _unscale(val: float, pad: float, ratio: float) -> float:
    return (val - pad) / ratio


# ── session wrapper ───────────────────────────────────────────────────────────

class YoloOnnxSession:
    """
    Thin wrapper around an onnxruntime InferenceSession for one YOLO model.

    providers — ordered list passed to onnxruntime, e.g.:
        ['DmlExecutionProvider', 'CPUExecutionProvider']    # AMD/Intel Windows
        ['CUDAExecutionProvider', 'CPUExecutionProvider']   # NVIDIA
        ['CoreMLExecutionProvider', 'CPUExecutionProvider'] # Apple Silicon
        ['CPUExecutionProvider']                            # fallback
    """

    def __init__(self, onnx_path: str, providers: list[str], names: dict[int, str] | None = None):
        import onnxruntime as ort
        self.session   = ort.InferenceSession(onnx_path, providers=providers)
        self.inp_name  = self.session.get_inputs()[0].name
        self.names     = names or {}

        # Class names embedded in the model always match its output indices.
        # Prefer them over an external names.json, which can drift out of sync
        # after a fine-tune deploy (model gains/loses classes but the json is
        # not regenerated → every char mislabeled → plates fail validation).
        try:
            import ast
            embedded = self.session.get_modelmeta().custom_metadata_map.get("names")
            if embedded:
                parsed = ast.literal_eval(embedded)
                resolved = {int(k): v for k, v in parsed.items()}
                if resolved:
                    if self.names and len(self.names) != len(resolved):
                        logger.warning(
                            "names.json has %d classes but model embeds %d — using model's embedded names",
                            len(self.names), len(resolved),
                        )
                    self.names = resolved
        except Exception as e:
            logger.warning("Could not read embedded class names from %s (%s) — using names.json",
                           os.path.basename(onnx_path), e)

        inp   = self.session.get_inputs()[0]
        # inp.shape may be [1, 3, H, W] or [batch, 3, H, W]
        self.imgsz = int(inp.shape[2]) if inp.shape[2] else 640

        out_shape = self.session.get_outputs()[0].shape
        # Post-NMS heuristic: last dim == 6 and second dim is small (≤ 1000 detections)
        self.is_post_nms = (
            len(out_shape) == 3
            and isinstance(out_shape[2], int) and out_shape[2] == 6
            and isinstance(out_shape[1], int) and out_shape[1] <= 1000
        )

        used = self.session.get_providers()
        logger.info(
            "ONNX session ready: %s  imgsz=%d  post_nms=%s  provider=%s",
            os.path.basename(onnx_path),
            self.imgsz,
            self.is_post_nms,
            used[0],
        )

    # ── public API ──────────────────────────────────────────────────────────

    def detect(
        self,
        img_bgr: np.ndarray,
        conf_thresh: float = 0.20,
        iou_thresh: float  = 0.45,
    ) -> list[dict]:
        """
        Returns list of dicts:
          { bbox: [x1,y1,x2,y2], confidence: float, class_id: int, class_name: str }
        Coordinates are in the original image pixel space.
        """
        orig_h, orig_w = img_bgr.shape[:2]
        sz = self.imgsz

        img_lb, ratio, (pad_w, pad_h) = _letterbox(img_bgr, (sz, sz))
        blob = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]          # 1CHW

        # Serialize GPU access with the rest of the process (face engine /
        # restorer). DirectML can crash or corrupt memory under concurrent
        # session.run() calls — onnx_infer is the plate-detection path that
        # was previously bypassing this lock.
        from app.gpu_lock import inference_lock
        with inference_lock:
            raw = self.session.run(None, {self.inp_name: blob})[0]  # (1, *, *)

        if self.is_post_nms:
            return self._decode_post_nms(raw[0], conf_thresh, orig_w, orig_h, ratio, pad_w, pad_h)
        else:
            return self._decode_pre_nms(raw[0], conf_thresh, iou_thresh, orig_w, orig_h, ratio, pad_w, pad_h)

    # ── decoders ────────────────────────────────────────────────────────────

    def _decode_post_nms(self, pred, conf_thresh, ow, oh, ratio, pad_w, pad_h):
        # pred: (max_det, 6)  →  [x1, y1, x2, y2, confidence, class_id]
        results = []
        for det in pred:
            conf = float(det[4])
            if conf < conf_thresh:
                continue
            cls_id = int(det[5])
            x1 = max(0, min(ow, _unscale(float(det[0]), pad_w, ratio)))
            y1 = max(0, min(oh, _unscale(float(det[1]), pad_h, ratio)))
            x2 = max(0, min(ow, _unscale(float(det[2]), pad_w, ratio)))
            y2 = max(0, min(oh, _unscale(float(det[3]), pad_h, ratio)))
            results.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
                "class_name": self.names.get(cls_id, str(cls_id)),
            })
        return results

    def _decode_pre_nms(self, pred, conf_thresh, iou_thresh, ow, oh, ratio, pad_w, pad_h):
        # pred: (4+nc, anchors) — transpose to (anchors, 4+nc)
        p = pred.T                          # (anchors, 4+nc)
        boxes_raw = p[:, :4]               # cx, cy, w, h
        scores    = p[:, 4:]               # (anchors, nc)

        cls_scores = scores.max(axis=1)
        cls_ids    = scores.argmax(axis=1)

        mask = cls_scores >= conf_thresh
        if not mask.any():
            return []

        boxes_xyxy = _xywh_to_xyxy(boxes_raw[mask])
        cls_s      = cls_scores[mask]
        cls_i      = cls_ids[mask]

        keep = _nms(boxes_xyxy, cls_s, iou_thresh)

        results = []
        for idx in keep:
            x1, y1, x2, y2 = boxes_xyxy[idx]
            conf   = float(cls_s[idx])
            cls_id = int(cls_i[idx])
            x1 = max(0, min(ow, _unscale(float(x1), pad_w, ratio)))
            y1 = max(0, min(oh, _unscale(float(y1), pad_h, ratio)))
            x2 = max(0, min(ow, _unscale(float(x2), pad_w, ratio)))
            y2 = max(0, min(oh, _unscale(float(y2), pad_h, ratio)))
            results.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
                "class_name": self.names.get(cls_id, str(cls_id)),
            })
        return results
