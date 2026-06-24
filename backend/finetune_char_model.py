"""
Fine-tune the Thai character YOLO model on approved CCTV samples merged with
local Roboflow datasets.

Fixed external datasets (configured via ROBOFLOW_DATASET_BASE env var):
  - Thai-License-Plate-Character-Recognition-10
  - Thai-LNPR-3
  - LRU-License-Plate-1
  - license-plate-charecter-5

All external datasets are remapped to MASTER_CLASSES before merging with CCTV data.

Usage (called by Go control plane):
    python3 finetune_char_model.py \
        --data /tmp/finetune_xxx/data.yaml \
        --base-model /path/to/thai_char_yolo26s.pt \
        --output /path/to/thai_char_yolo26s.pt \
        --epochs 30

Prints JSON lines to stdout:
    {"type": "epoch", "epoch": 5, "epochs": 30, "box_loss": 0.12}
    {"type": "done", "model": "...", "onnx": "...", "version": "..."}
    {"type": "error", "message": "..."}
"""

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile

import yaml


# ── Master class list (129 classes, authoritative) ────────────────────────────

MASTER_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A01', 'A02', 'A04', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12',
    'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22',
    'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32',
    'A33', 'A34', 'A35', 'A36', 'A37', 'A38', 'A39', 'A40', 'A41', 'A42',
    'A43', 'A44',
    'ACR', 'ATG', 'AYA', 'BKK', 'BKN', 'BRM', 'CBI', 'CCO', 'CMI', 'CNT',
    'CPM', 'CPN', 'CRI', 'CTI', 'KBI', 'KKN', 'KPT', 'KRI', 'KSN', 'LEI',
    'LPG', 'LPN', 'LRI', 'MDH', 'MKM', 'NAN', 'NBI', 'NBP', 'NKI', 'NMA',
    'NPM', 'NPT', 'NRT', 'NSN', 'NST', 'NWT', 'NYK', 'PBI', 'PCT', 'PKN',
    'PKT', 'PLG', 'PLK', 'PNA', 'PNB', 'PRE', 'PRI', 'PTE', 'PTN', 'PYO',
    'RBR', 'RET', 'RNG', 'RYG', 'SBR', 'SKA', 'SKM', 'SKN', 'SKW', 'SNI',
    'SNK', 'SPB', 'SPK', 'SRI', 'SRN', 'SSK', 'STI', 'STN', 'TAK', 'TRG',
    'TRT', 'UBN', 'UDN', 'UTI', 'UTT', 'YLA', 'YST',
]

MASTER_INDEX = {name: i for i, name in enumerate(MASTER_CLASSES)}

# Global aliases: common typos / Roboflow defaults → canonical MASTER name
GLOBAL_ALIASES = {
    'BK.': 'BKK', 'CHAN': 'CNT', 'CHM': 'CMI', 'CHON': 'CBI',
    'CHP': 'CPM', 'CHR': 'CRI', 'CYP': 'CPN',
    'KAL': 'KSN', 'KHON': 'KKN', 'KOR': 'KRI', 'KPP': 'KPT',
    'LAMP': 'LPG', 'LB': 'LRI', 'LP': 'LPN',
    'MHK': 'MKM', 'MHS': 'MDH', 'PY': 'PYO', 'RE': 'RET',
    'SRTN': 'SRN', 'SUPB': 'SPB', 'SUR': 'SRI',
    'UBON': 'UBN', 'UDON': 'UDN', 'na': 'NAN',
    'N0': '0', 'N1': '1', 'N2': '2', 'N3': '3', 'N4': '4',
    'N5': '5', 'N6': '6', 'N7': '7', 'N8': '8', 'N9': '9',
    'N13': '1', 'N23': '2',
}

# Per-dataset label aliases (folder name → {source_label: canonical_label})
DATASET_SPECIFIC_ALIASES = {
    "Thai-License-Plate-Character-Recognition-10": {
        # KMITL dataset uses shifted A## labels
        'A11': 'A11', 'A12': 'A11', 'A13': 'A12', 'A14': 'A13', 'A16': 'A14',
        'A18': 'A16', 'A19': 'A17', 'A20': 'A18', 'A21': 'A19', 'A22': 'A20',
        'A23': 'A21', 'A24': 'A22', 'A25': 'A23', 'A26': 'A24', 'A27': 'A25',
        'A28': 'A26', 'A30': 'A27', 'A31': 'A28', 'A32': 'A29', 'A33': 'A30',
        'A34': 'A31', 'A35': 'A32', 'A36': 'A33', 'A37': 'A34', 'A38': 'A35',
        'A39': 'A36', 'A40': 'A37', 'A41': 'A38', 'A42': 'A39', 'A43': 'A40',
        'A44': 'A41',
    },
    "license-plate-charecter-5": {
        'A11': 'A11', 'A12': 'A11', 'A13': 'A12', 'A14': 'A13', 'A16': 'A14',
        'A18': 'A16', 'A19': 'A17', 'A20': 'A18', 'A21': 'A19', 'A22': 'A20',
        'A23': 'A21', 'A24': 'A22', 'A25': 'A23', 'A26': 'A24', 'A27': 'A25',
        'A28': 'A26', 'A30': 'A27', 'A31': 'A28', 'A32': 'A29', 'A33': 'A30',
        'A34': 'A31', 'A35': 'A32', 'A36': 'A33', 'A37': 'A34', 'A38': 'A35',
        'A39': 'A36', 'A40': 'A37', 'A41': 'A38', 'A42': 'A39', 'A43': 'A40',
        'A44': 'A41',
    },
    "LRU-License-Plate-1": {
        'A1': 'A01', 'A2': 'A02', 'A4': 'A04', 'A6': 'A06',
        'A7': 'A07', 'A8': 'A08', 'A9': 'A09', 'A10': 'A10',
        'a40': 'A40', 'a9.': 'A10', 'ANC': 'ACR', 'NSA': 'NSN',
    },
    "Thai-LNPR-3": {
        '10GorGai': 'A01', '11KhorKhai': 'A02', '13KhorKhwai': 'A04',
        '15KhorRaKhang': 'A06', '16NgorNgu': 'A07',
        '17ChorChan': 'A08', '18ChorChing': 'A09',
        '19ChorChang': 'A10', '20SorSo': 'A11', '21ChorChoer': 'A12',
        '22YorYing': 'A13', '23DorChaDa': 'A14', '24TorPaTak': 'A15',
        '25ThorThan': 'A16', '26ThorMontho': 'A17', '27ThorPhuThao': 'A18',
        '28NorNen': 'A19', '29DorDek': 'A20', '30TorTao': 'A21',
        '31ThorThung': 'A22', '32ThorThaHan': 'A23', '33ThorThong': 'A24',
        '34NorNu': 'A25', '35BorBaiMai': 'A26', '36PorPla': 'A27',
        '37PhorPhueng': 'A28', '38ForFa': 'A29', '39PhorPan': 'A30',
        '40ForFan': 'A31', '41PhorSamPhao': 'A32', '42MorMa': 'A33',
        '43YorYak': 'A34', '44RorRuea': 'A35', '45LorLing': 'A36',
        '46WorWaen': 'A37', '47SorSala': 'A38', '48SorRueSi': 'A39',
        '49SorSuea': 'A40', '50HorHeep': 'A41', '51LorJuLa': 'A42',
        '52OrAng': 'A43', '53HorNokHook': 'A44',
        'AngThong': 'ATG', 'Ayutthaya': 'AYA', 'Bangkok': 'BKK',
        'BuengKan': 'BKN', 'Buriram': 'BRM',
        'Chachoengsao': 'CCO', 'ChaiNat': 'CNT', 'Chaiyaphum': 'CPM',
        'Chanthaburi': 'CTI', 'ChiangMai': 'CMI', 'ChiangRai': 'CRI',
        'Chonburi': 'CBI', 'Chumphon': 'CPN',
        'Kalasin': 'KSN', 'KamphaengPhet': 'KPT', 'Kanchanaburi': 'KRI',
        'KhonKaen': 'KKN', 'Krabi': 'KBI',
        'Lampang': 'LPG', 'Lamphun': 'LPN', 'Loei': 'LEI', 'LopBuri': 'LRI',
        'MahaSarakham': 'MDH', 'Mukdahan': 'MKM',
        'NakhonNayok': 'NYK', 'NakhonPathom': 'NPM',
        'NakhonRatchasima': 'NMA', 'NakhonSawan': 'NSN',
        'NongKhai': 'NKI', 'Nonthaburi': 'NBP',
        'PathumThani': 'PRI', 'Phetchabun': 'PNB',
        'Phichit': 'PCT', 'Phitsanulok': 'PLK', 'Phuket': 'PKT',
        'PrachinBuri': 'PBI', 'PrachuapKhiriKhan': 'PKN',
        'Ratchaburi': 'RBR', 'Rayong': 'RYG', 'RoiEt': 'RET',
        'SaKaeo': 'SKW', 'SakonNakhon': 'SNK',
        'SamutPrakan': 'SPK', 'SamutSakhon': 'SKN', 'SamutSongkhram': 'SKM',
        'SaraBuri': 'SBR', 'SingBuri': 'SNI', 'Songkhla': 'SKA',
        'Sukhothai': 'STN', 'SuphanBuri': 'SPB',
        'SuratThani': 'SRN', 'Surin': 'SRI',
        'Tak': 'TAK', 'Trang': 'TRG',
        'UbonRatchathani': 'UBN', 'UdonThani': 'UDN', 'UthaiThani': 'UTI',
        'Yala': 'YLA', 'zBayTong': 'PTN', 'zPhangNga': 'PNA',
    },
}

# Fixed Roboflow datasets to merge (folder names under ROBOFLOW_DATASET_BASE)
FIXED_DATASETS = [
    "Thai-License-Plate-Character-Recognition-10",
    "Thai-LNPR-3",
    "LRU-License-Plate-1",
    "license-plate-charecter-5",
]

ROBOFLOW_SOURCES = [
    {"folder": "Thai-License-Plate-Character-Recognition-10", "workspace": "meenyossakorn",          "project": "thai-license-plate-character-recognition", "version": 10},
    {"folder": "license-plate-charecter-5",                   "workspace": "mydataset-zrfok",        "project": "license-plate-charecter",                  "version": 5},
    {"folder": "LRU-License-Plate-1",                         "workspace": "lru",                    "project": "lru-license-plate",                        "version": 1},
    {"folder": "Thai-LNPR-3",                                 "workspace": "thai-car-detection-mboy6","project": "thai-lnpr-c6prf",                          "version": 3},
]


def download_roboflow_datasets(base_dir: str, api_key: str) -> None:
    """Download any missing Roboflow datasets into base_dir. Skips already-downloaded ones."""
    if not api_key:
        return
    try:
        from roboflow import Roboflow
    except ImportError:
        emit({"type": "info", "message": "roboflow package not installed — skipping auto-download (pip install roboflow)"})
        return

    rf = Roboflow(api_key=api_key)
    for ds in ROBOFLOW_SOURCES:
        dest = os.path.join(base_dir, ds["folder"])
        # Consider already downloaded if the train/images subfolder exists and has files
        img_dir = os.path.join(dest, "train", "images")
        if os.path.isdir(img_dir) and any(True for _ in os.scandir(img_dir)):
            emit({"type": "info", "message": f"[roboflow] {ds['folder']} already exists — skipping"})
            continue
        emit({"type": "info", "message": f"[roboflow] Downloading {ds['folder']} ..."})
        try:
            project = rf.workspace(ds["workspace"]).project(ds["project"])
            project.version(ds["version"]).download("yolov8", location=dest)
            emit({"type": "info", "message": f"[roboflow] {ds['folder']} downloaded"})
        except Exception as e:
            emit({"type": "info", "message": f"[roboflow] {ds['folder']} download failed: {e}"})


# ── Helpers ───────────────────────────────────────────────────────────────────

_progress_callback = None  # set to a callable(dict) when used inline
_stop_event = None         # threading.Event set externally to abort training

def emit(obj: dict):
    if _progress_callback is not None:
        _progress_callback(obj)
    else:
        print(json.dumps(obj), flush=True)


def label_to_bbox(parts: list) -> str | None:
    """Normalise a label line to 5-field YOLO bbox format."""
    if len(parts) == 5:
        return " ".join(parts)
    if len(parts) > 5:
        try:
            coords = [float(x) for x in parts[1:]]
            xs, ys = coords[0::2], coords[1::2]
            xc = (min(xs) + max(xs)) / 2
            yc = (min(ys) + max(ys)) / 2
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            return f"{parts[0]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        except ValueError:
            pass
    return None


def remap_labels(ds_dir: str, ds_folder: str, prefix: str, out_dir: str) -> tuple[int, int]:
    """
    Copy images and remap labels from ds_dir → out_dir/{train,valid,test}.
    Returns (copied_count, skipped_labels).
    """
    yaml_path = os.path.join(ds_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        return 0, 0

    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    src_names = cfg.get("names", [])

    specific = DATASET_SPECIFIC_ALIASES.get(ds_folder, {})

    # Build remap: source class index → master class index
    remap: dict[int, int] = {}
    for old_id, name in enumerate(src_names):
        canonical = specific.get(name, GLOBAL_ALIASES.get(name, name))
        if canonical in MASTER_INDEX:
            remap[old_id] = MASTER_INDEX[canonical]

    copied = 0
    skipped = 0

    for split in ("train", "valid", "test"):
        src_img = os.path.join(ds_dir, split, "images")
        src_lbl = os.path.join(ds_dir, split, "labels")
        dst_img = os.path.join(out_dir, split, "images")
        dst_lbl = os.path.join(out_dir, split, "labels")
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_lbl, exist_ok=True)

        if not os.path.exists(src_img):
            continue

        for fname in os.listdir(src_img):
            shutil.copy2(os.path.join(src_img, fname),
                         os.path.join(dst_img, prefix + fname))
            copied += 1

            stem = os.path.splitext(fname)[0]
            lbl_src = os.path.join(src_lbl, stem + ".txt")
            lbl_dst = os.path.join(dst_lbl, prefix + stem + ".txt")
            if not os.path.exists(lbl_src):
                open(lbl_dst, "w").close()
                continue

            with open(lbl_src) as fh:
                lines = fh.readlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    old_id = int(parts[0])
                    if old_id in remap:
                        parts[0] = str(remap[old_id])
                        row = label_to_bbox(parts)
                        if row:
                            new_lines.append(row)
                    else:
                        skipped += 1
                except (ValueError, IndexError):
                    skipped += 1
            with open(lbl_dst, "w") as fh:
                fh.write("\n".join(new_lines) + ("\n" if new_lines else ""))

    return copied, skipped


def merge_roboflow_datasets(base_dir: str, out_dir: str) -> str | None:
    """
    Merge all FIXED_DATASETS found under base_dir into out_dir with MASTER_CLASSES.
    Returns path to merged data.yaml, or None if no datasets found.
    """
    found = 0
    for i, folder in enumerate(FIXED_DATASETS, start=1):
        ds_path = os.path.join(base_dir, folder)
        if not os.path.exists(ds_path):
            emit({"type": "info", "message": f"[datasets] {folder} not found — skipping"})
            continue
        copied, skipped = remap_labels(ds_path, folder, f"ds{i}_", out_dir)
        emit({"type": "info", "message": f"[datasets] {folder}: {copied} images, {skipped} labels skipped"})
        found += 1

    if found == 0:
        return None

    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump({
            "nc": len(MASTER_CLASSES),
            "names": MASTER_CLASSES,
            "train": os.path.abspath(os.path.join(out_dir, "train/images")),
            "val":   os.path.abspath(os.path.join(out_dir, "valid/images")),
        }, f, default_flow_style=False, allow_unicode=True)
    return yaml_path


def build_combined_yaml(cctv_yaml: str, ext_yaml: str, out_dir: str) -> str:
    """
    Combine CCTV data.yaml + external data.yaml into a single multi-path yaml.
    Both must use MASTER_CLASSES (same nc/names).
    """
    with open(cctv_yaml, encoding="utf-8") as f:
        cctv = yaml.safe_load(f)
    with open(ext_yaml, encoding="utf-8") as f:
        ext = yaml.safe_load(f)

    def as_list(v, base):
        if not v:
            return []
        items = v if isinstance(v, list) else [v]
        return [p if os.path.isabs(p) else os.path.join(base, p) for p in items]

    cctv_dir = os.path.dirname(os.path.abspath(cctv_yaml))
    ext_dir  = os.path.dirname(os.path.abspath(ext_yaml))

    combined = {
        "nc":    cctv["nc"],
        "names": cctv["names"],
        "train": as_list(cctv.get("train"), cctv_dir) + as_list(ext.get("train"), ext_dir),
        "val":   as_list(cctv.get("val", cctv.get("valid")), cctv_dir) +
                 as_list(ext.get("val",  ext.get("valid")),  ext_dir),
    }
    out = os.path.join(out_dir, "combined_data.yaml")
    with open(out, "w", encoding="utf-8") as f:
        yaml.dump(combined, f, default_flow_style=False, allow_unicode=True)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        required=True, help="CCTV data.yaml from Go export")
    parser.add_argument("--base-model",  required=True, help="Base .pt model path")
    parser.add_argument("--output",      required=True, help="Output .pt model path")
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--imgsz",       type=int, default=640)
    parser.add_argument("--batch",       type=int, default=8)
    parser.add_argument("--samples",     type=int, default=0, help="Approved sample count (metadata)")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        emit({"type": "error", "message": f"data.yaml not found: {args.data}"}); sys.exit(1)
    if not os.path.exists(args.base_model):
        emit({"type": "error", "message": f"Base model not found: {args.base_model}"}); sys.exit(1)

    try:
        from ultralytics import YOLO
        import torch
    except ImportError as e:
        emit({"type": "error", "message": f"Import error: {e}"}); sys.exit(1)

    tmp_root = tempfile.mkdtemp(prefix="finetune_")
    try:
        _run(args, tmp_root, YOLO, torch)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _run(args, tmp_root, YOLO, torch):
    import datetime

    # ── 1. Merge Roboflow datasets with label remapping ───────────────────────
    # Default: datasets/ folder next to the models dir (e.g. data/datasets/)
    _models_dir = os.path.dirname(os.path.abspath(args.output))
    _default_base = os.path.join(os.path.dirname(_models_dir), "datasets")
    roboflow_base = os.environ.get("ROBOFLOW_DATASET_BASE", _default_base)
    ext_dir = os.path.join(tmp_root, "ext_datasets")
    os.makedirs(ext_dir, exist_ok=True)

    emit({"type": "info", "message": f"Merging Roboflow datasets from {roboflow_base} ..."})
    ext_yaml = merge_roboflow_datasets(roboflow_base, ext_dir)

    # ── 2. Combine with CCTV data ─────────────────────────────────────────────
    if ext_yaml:
        data_yaml = build_combined_yaml(args.data, ext_yaml, tmp_root)
        emit({"type": "info", "message": "Combined CCTV + Roboflow datasets ready"})
    else:
        data_yaml = args.data
        emit({"type": "info", "message": "No external datasets found — training on CCTV only"})

    total_epochs = args.epochs

    # ── 3. Device selection ───────────────────────────────────────────────────
    device = os.environ.get("TRAIN_DEVICE", "").strip()
    if not device:
        if torch.cuda.is_available():
            device = "0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            try:
                import torch_directml
                device = torch_directml.device()
                emit({"type": "info", "message": "Trying DirectML (AMD/Intel GPU) for training"})
            except ImportError:
                device = "cpu"

    emit({"type": "info", "message": f"Training on device={device!r}, epochs={total_epochs}"})

    # ── 4. Train ──────────────────────────────────────────────────────────────
    # Use tmp_root as project dir so runs go there, not the process cwd.
    # Use a fixed name so best.pt path is deterministic.
    train_project = os.path.join(tmp_root, "runs")
    train_name    = "finetune_char"
    best_pt_path  = os.path.join(train_project, train_name, "weights", "best.pt")

    def on_train_epoch_end(trainer):
        loss_names = getattr(trainer, "loss_names", [])
        losses     = getattr(trainer, "loss_items", None)
        loss_dict  = {}
        if losses is not None and loss_names:
            for name, val in zip(loss_names, losses):
                try:
                    loss_dict[name] = round(float(val), 5)
                except Exception:
                    pass
        emit({"type": "epoch", "epoch": trainer.epoch + 1, "epochs": total_epochs, **loss_dict})
        if _stop_event is not None and _stop_event.is_set():
            emit({"type": "info", "message": "Stop requested — aborting after this epoch"})
            trainer.stop = True

    # In a PyInstaller frozen build, dataloader subprocesses re-launch the exe,
    # which is unsafe. Load data in the main process (workers=0) when frozen.
    train_workers = 0 if getattr(sys, "frozen", False) else 2

    def run_train(dev):
        # AMP (mixed precision) is CUDA-only in ultralytics. Its startup AMP check
        # calls torch.cuda and crashes on DirectML/CPU ("Torch not compiled with
        # CUDA enabled"), so only enable it on a real CUDA GPU.
        use_amp = (
            torch.cuda.is_available()
            and isinstance(dev, str)
            and dev not in ("cpu", "mps")
        )
        model = YOLO(args.base_model)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        try:
            model.train(
                data=data_yaml,
                epochs=total_epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=dev,
                project=train_project,
                name=train_name,
                exist_ok=True,
                patience=10,
                save=True,
                verbose=False,
                workers=train_workers,
                cache=False,
                plots=False,
                amp=use_amp,
            )
        finally:
            # Release GPU memory before returning (critical for DirectML — no empty_cache()).
            # ultralytics may hold trainer references internally; delete model explicitly.
            try:
                del model.trainer
            except Exception:
                pass
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    try:
        run_train(device)
    except Exception as e:
        # DirectML / non-CUDA GPU training under ultralytics is unreliable.
        # Fall back to CPU so the job still completes (slower but works).
        if device != "cpu":
            emit({"type": "info", "message": f"GPU training failed ({e}) — retrying on CPU"})
            try:
                run_train("cpu")
            except Exception as e2:
                emit({"type": "error", "message": f"CPU training also failed: {e2}"}); return
        else:
            emit({"type": "error", "message": str(e)}); return

    # ── 5. Save best.pt ───────────────────────────────────────────────────────
    if not os.path.exists(best_pt_path):
        emit({"type": "error", "message": f"best.pt not found at {best_pt_path}"}); sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    shutil.copy2(best_pt_path, args.output)
    emit({"type": "info", "message": f"Saved .pt → {args.output}"})

    # ── 6. Export ONNX ────────────────────────────────────────────────────────
    onnx_output = args.output.replace(".pt", ".onnx")
    emit({"type": "info", "message": "Exporting to ONNX..."})
    # onnxsim (simplify=True) spawns a subprocess — in a PyInstaller frozen exe
    # that re-launches the GUI. Disable simplify when frozen; skip it otherwise
    # too since the accuracy/size benefit is minor for inference-only models.
    is_frozen = getattr(sys, "frozen", False)
    try:
        YOLO(args.output).export(format="onnx", imgsz=args.imgsz, opset=12,
                                 simplify=not is_frozen, dynamic=False, half=False)
        alongside = args.output.replace(".pt", ".onnx")
        if os.path.exists(alongside) and alongside != onnx_output:
            shutil.move(alongside, onnx_output)
        emit({"type": "info", "message": f"Saved .onnx → {onnx_output}"})
    except Exception as e:
        emit({"type": "info", "message": f"ONNX export failed ({e}) — .pt saved, engine will fallback"})

    # ── 7. Save versioned backup ──────────────────────────────────────────────
    version_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    versions_dir = os.path.join(os.path.dirname(args.output), "versions", version_ts)
    os.makedirs(versions_dir, exist_ok=True)
    shutil.copy2(args.output, os.path.join(versions_dir, os.path.basename(args.output)))
    if os.path.exists(onnx_output):
        shutil.copy2(onnx_output, os.path.join(versions_dir, os.path.basename(onnx_output)))
    with open(os.path.join(versions_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "version":    version_ts,
            "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
            "samples":    args.samples,
            "epochs":     args.epochs,
            "base_model": os.path.basename(args.base_model),
        }, f, indent=2)
    emit({"type": "info", "message": f"Version saved → {versions_dir}"})

    emit({"type": "done", "model": args.output, "onnx": onnx_output, "version": version_ts})


def run_finetune_inline(
    base_model: str,
    output_model: str,
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 8,
    cctv_yaml: str | None = None,
    roboflow_base: str | None = None,
    progress_cb=None,
    stop_event=None,
    roboflow_api_key: str = "",
):
    """
    Call fine-tuning directly in-process (no subprocess).
    progress_cb(dict) is called for each progress event instead of printing to stdout.
    Used by the PyInstaller Windows worker.
    """
    import types as _types

    global _progress_callback
    _progress_callback = progress_cb

    if roboflow_base:
        os.environ.setdefault("ROBOFLOW_DATASET_BASE", roboflow_base)

    try:
        from ultralytics import YOLO
        import torch
    except ImportError as e:
        emit({"type": "error", "message": f"Import error: {e}"})
        return

    args = _types.SimpleNamespace(
        data=cctv_yaml or "",
        base_model=base_model,
        output=output_model,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        samples=0,
    )

    if args.data and not os.path.exists(args.data):
        emit({"type": "error", "message": f"data.yaml not found: {args.data}"})
        return
    if not os.path.exists(args.base_model):
        emit({"type": "error", "message": f"Base model not found: {args.base_model}"})
        return

    global _stop_event
    _stop_event = stop_event

    if roboflow_api_key and roboflow_base:
        os.makedirs(roboflow_base, exist_ok=True)
        download_roboflow_datasets(roboflow_base, roboflow_api_key)

    tmp_root = tempfile.mkdtemp(prefix="finetune_")
    try:
        _run(args, tmp_root, YOLO, torch)
    except Exception as e:
        emit({"type": "error", "message": str(e)})
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        _progress_callback = None
        _stop_event = None


if __name__ == "__main__":
    main()
