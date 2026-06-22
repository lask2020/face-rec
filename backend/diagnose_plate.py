"""
Diagnose why license-plate detection is not working.

Run on the ACTUAL worker machine, inside its venv:
    Windows:  venv_native\\Scripts\\python backend\\diagnose_plate.py
    (run from the project root)

It checks, in order:
  1. onnxruntime flavour + available providers (DirectML for AMD GPU)
  2. model files present
  3. LicensePlateEngine loads and which provider it actually uses
  4. a real detect() pass on a test image (optional: pass an image path)
"""
import os
import sys

# Make `app` importable regardless of where this is launched from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  LICENSE PLATE DIAGNOSTIC")
print("=" * 60)

# 1. onnxruntime ------------------------------------------------------------
print("\n[1] onnxruntime")
try:
    import onnxruntime as ort
    print("    version :", ort.__version__)
    provs = ort.get_available_providers()
    print("    providers:", provs)
    if "DmlExecutionProvider" not in provs:
        print("    !! DmlExecutionProvider NOT available.")
        print("       For Windows AMD GPU you need onnxruntime-directml:")
        print("       pip uninstall -y onnxruntime onnxruntime-gpu")
        print("       pip install onnxruntime-directml")
except Exception as e:
    print("    !! onnxruntime import FAILED:", e)
    print("       pip install onnxruntime-directml")

# 2. model files ------------------------------------------------------------
print("\n[2] model files")
from app.license_plate.engine import _MODELS_DIR
print("    models dir:", os.path.abspath(_MODELS_DIR))
for fn in ("thai_plate_yolo11n.onnx", "thai_char_yolo26s.onnx",
           "thai_char_yolo26s_names.json"):
    p = os.path.join(_MODELS_DIR, fn)
    print(f"    {'OK ' if os.path.exists(p) else 'MISSING'} {fn}")

# 3. engine load ------------------------------------------------------------
print("\n[3] engine load")
import logging
logging.basicConfig(level=logging.INFO, format="    %(levelname)s %(message)s")
from app.license_plate import LicensePlateEngine
eng = LicensePlateEngine()
print("    READY   :", eng.ready)
print("    use_onnx:", eng._use_onnx)
if eng.ready and eng._use_onnx:
    print("    provider:", eng._plate_model.session.get_providers()[0])
if not eng.ready:
    print("    !! Engine not ready — plate detection is DISABLED in the worker.")
    sys.exit(1)

# 4. detect pass ------------------------------------------------------------
print("\n[4] detect() pass")
import numpy as np
import cv2
img_path = sys.argv[1] if len(sys.argv) > 1 else None
if img_path and os.path.exists(img_path):
    frame = cv2.imread(img_path)
    print("    image:", img_path, frame.shape)
else:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    print("    no image given — using blank frame (expect 0 plates).")
    print("    Pass a real photo:  python backend/diagnose_plate.py path\\to\\car.jpg")

res = eng.detect(frame)
print(f"    detected {len(res)} plate(s)")
for r in res:
    print(f"      raw='{r.raw_text}'  plate={r.plate_number}  conf={r.confidence:.2f}  province={r.province}")

print("\nDone.")
