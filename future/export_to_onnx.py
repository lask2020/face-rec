"""
Export Thai license plate YOLO models (.pt) to ONNX format.

Output ONNX models are compatible with:
  - onnxruntime (CPU)
  - onnxruntime-directml  (AMD / Intel GPU on Windows)
  - onnxruntime-gpu       (NVIDIA GPU)
  - CoreMLExecutionProvider (Apple Silicon)

Usage:
    python future/export_to_onnx.py

Output:
    backend/models/thai_plate_yolo11n.onnx
    backend/models/thai_char_yolo26s.onnx
"""

import os
import sys

# Run from project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "backend", "models")


def export(pt_name: str, imgsz: int, opset: int = 12) -> str:
    from ultralytics import YOLO

    pt_path = os.path.join(MODELS_DIR, pt_name)
    if not os.path.exists(pt_path):
        print(f"[SKIP] {pt_path} not found")
        return ""

    onnx_name = pt_name.replace(".pt", ".onnx")
    onnx_path = os.path.join(MODELS_DIR, onnx_name)

    print(f"\n{'='*60}")
    print(f"Exporting: {pt_name}  →  {onnx_name}")
    print(f"  imgsz={imgsz}, opset={opset}")
    print(f"{'='*60}")

    model = YOLO(pt_path)
    model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=True,   # onnx-simplifier: removes redundant nodes
        dynamic=False,   # fixed batch=1 for DirectML compatibility
        half=False,      # FP32 — DirectML works better with FP32
    )

    # ultralytics exports to same directory as the .pt file
    exported = pt_path.replace(".pt", ".onnx")
    if os.path.exists(exported) and exported != onnx_path:
        import shutil
        shutil.move(exported, onnx_path)

    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"  Saved: {onnx_path} ({size_mb:.1f} MB)")
    return onnx_path


def verify(onnx_path: str, imgsz: int):
    """Quick sanity check — run one dummy inference."""
    import numpy as np
    import onnxruntime as ort

    print(f"\nVerifying {os.path.basename(onnx_path)} ...")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    out = sess.run(None, {inp_name: dummy})
    print(f"  Input:  {inp_name} {dummy.shape}")
    print(f"  Output: {[o.shape for o in out]}")
    print(f"  OK")


if __name__ == "__main__":
    # Plate detection model  — higher resolution for finding plates in full frames
    plate_onnx = export("thai_plate_yolo11n.pt", imgsz=1280)

    # Character recognition model — standard 640 resolution
    char_onnx = export("thai_char_yolo26s.pt", imgsz=640)

    print("\n\nRunning inference verification...")
    if plate_onnx and os.path.exists(plate_onnx):
        verify(plate_onnx, 1280)
    if char_onnx and os.path.exists(char_onnx):
        verify(char_onnx, 640)

    print("\nDone! To use DirectML on Windows AMD GPU:")
    print("  pip uninstall onnxruntime")
    print("  pip install onnxruntime-directml")
    print("  Then set ONNX_PROVIDER=DmlExecutionProvider in environment")
