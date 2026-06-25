# face-rec — Project Context for Claude

## Architecture Overview

```
frontend/          React + TypeScript (Vite)
go-control-plane/  Go (Fiber) — REST API, gRPC server, DB, S3
backend/           Python — AI worker (face + plate detection)
```

**Communication flow:**
- Go control plane ↔ Python AI worker via **gRPC bidirectional stream** (`facerec.proto`)
- Go exposes REST API at `/api/*`, consumed by React frontend
- Images stored in **MinIO S3** (`SnapshotsBucket`)
- DB: **PostgreSQL** via GORM

---

## Proto

Root: `facerec.proto` — **source of truth**. `backend/facerec.proto` is a copy.

When you change the proto, regenerate **both**:
```bash
# Python (from repo root)
venv_native/bin/python3 -m grpc_tools.protoc \
  --python_out=backend --grpc_python_out=backend -I. facerec.proto

# Go (from repo root)
protoc --go_out=go-control-plane --go-grpc_out=go-control-plane -I. facerec.proto

# Then sync the copy
cp facerec.proto backend/facerec.proto
```
`protoc` is at `/opt/homebrew/bin/protoc` (libprotoc 35.0).
`venv_native/` contains the Python venv with `grpc_tools`.

---

## License Plate Pipeline

### Detection engine (`backend/app/license_plate/engine.py`)
- Stage 1: `thai_plate_yolo11n` — detect plate bounding box
- Stage 2: `thai_char_yolo26s` — detect individual characters inside crop
- ONNX Runtime inference; falls back to `.pt` (ultralytics)
- Preprocessing: CLAHE grayscale for ONNX, color for bbox storage
- `CharDet` fields: `x_norm, char, confidence, cx, cy, bw, bh` (YOLO-normalized in deskewed crop space)
- `PlateResult.crop_bytes` — JPEG of deskewed color crop (for training data)
- `_assemble_chars(crop_shape=...)` — builds `CharDet` list with YOLO bbox from raw padded-image coords

### PlateTrack (`backend/ai_worker_grpc.py`)
- Spatial bbox matching: IoU ≥ `PLATE_IOU_THRESH` (0.4) or center-distance fallback
- Accumulates **all frames** in `frame_results: list[PlateResult]`
- Flushed on timeout (`PLATE_TRACK_TIMEOUT=6s`) or max duration (`PLATE_TRACK_MAX_DURATION=12s`)
- `_assemble_multi_frame()` — combines all frames of a track into one read:
  - **Step 0 (quality gate, added):** drop structurally-implausible frames *before* voting —
    keep only frames with char count 2..8, and if any frame saw a Thai consonant, discard
    the consonant-less (pure-digit) frames as detector noise. Prevents garbage frames
    (all-digit / absurd-length like `377131565`) from polluting the vote.
  - **Count group** chosen by **highest total confidence** (not most frames) — a few
    high-conf reads of the right length beat many low-conf garbage frames.
  - Per-slot best confidence across the chosen group, province by majority vote.
- **Flush gate (`flush_plate_track`)**: a track is only sent to the control plane if it
  parsed into a **valid `plate_number`**. Unparseable reads (`plate_number is None` — pure-digit
  junk, partial reads like `22ก`, wrong-length garbage) are dropped entirely *regardless of
  confidence*, since a real Thai plate always has ≥2 consonants + a number block. This keeps the
  detection log clean. Training-frame capture is independent of this gate (keys off
  `TRAINING_CAPTURE_CONF_MIN`), so dropping invalid reads costs **no** training data.
- **Fuzzy cooldown**: Levenshtein distance ≤ `PLATE_FUZZY_DIST` (default 2) prevents duplicate DB records from OCR-variant plates

### Plate text validation (`backend/app/license_plate/validate.py`)
- `validate()` → `(is_valid, normalized, conf)` matches old/new/motorcycle Thai formats (requires a real hyphen + ≥2 consonants).
- `correct_common_errors()` — **conservative only**: hyphen recovery + a few literal artifact
  fixes (`เ→4`, `B→8`, `l/I→1`) that all require two real Thai consonants to already be present.
- **No consonant fabrication**: the old `_map_to_thai` / `SHAPE_CANDIDATES` digit→consonant
  guesser was **removed** — it invented plausible-but-wrong plates from digit-only OCR reads
  (e.g. `133327` → `1-รร-327` at conf 0.87). If the OCR reads a consonant slot as a number,
  the read is dropped (`plate_number=None`) rather than guessed.
- **Confidence penalty** (`PLATE_CORRECTION_CONF_PENALTY`, default 0.8): applied in BOTH
  `engine.py` and `_assemble_multi_frame` when a plate only validated *after* a real
  character-changing correction (hyphen-only insertion is cosmetic and NOT penalized), so
  corrected reads never outrank clean reads when sorting/filtering by confidence.

> **`plate_detection_logs` vs `plate_training_samples`**: detection logs store the
> *assembled + validated* read (`plate_number` normalized, `raw_text` pre-validation);
> training samples store *per-frame raw OCR* (no validator, no correction) — which is why
> training data looks "cleaner" (it's what the model actually saw).

### Key env vars (Python worker)
| Var | Default | Purpose |
|-----|---------|---------|
| `PLATE_FUZZY_DIST` | `2` | Max edit distance for duplicate suppression |
| `PLATE_TRACK_TIMEOUT` | `6.0` | Seconds before flushing idle track |
| `PLATE_TRACK_MAX_DURATION` | `12.0` | Max seconds to accumulate a track |
| `MIN_PLATE_HITS` | `1` | Minimum frames required to flush |
| `PLATE_CORRECTION_CONF_PENALTY` | `0.8` | Confidence multiplier when a plate validated only after a real OCR-artifact correction |
| `TRAINING_CAPTURE_CONF_MIN` | `0.45` | Frames at/above this conf are captured for training review |
| `TRAINING_MAX_FRAMES_PER_TRACK` | `3` | Max training frames sent per track |

---

## Training / Active-Learning Pipeline

Low-confidence plate crops are automatically captured and saved for OCR retraining.

### Data flow
1. Python worker: frames with `confidence >= TRAINING_CAPTURE_CONF_MIN` (and char-level data) → `PlateTrainingFrame` in gRPC `InferenceResult` (top-N by confidence, capped at `TRAINING_MAX_FRAMES_PER_TRACK`)
2. Go: `saveTrainingFrames()` uploads JPEG to S3 (`training_cam{id}_{ts}_{idx}.jpg`), inserts `PlateTrainingSample` row
3. Frontend `/training` page: review grid → approve / reject / correct text
4. Export ZIP: YOLO-format dataset (train 90% / valid 10%), `data.yaml` with 129 `MASTER_CLASSES`

### DB model: `PlateTrainingSample` (table `plate_training_samples`)
Fields: `id, camera_id, camera_name, image_path (S3 key), char_labels (JSON), raw_text, corrected_text, confidence, status (pending|approved|rejected), detected_at, created_at`

### REST endpoints (Go)
```
GET    /api/training/plates               list (active-learning sort: conf ASC)
GET    /api/training/plates/stats         by_status + by_class counts
GET    /api/training/plates/export        download YOLO ZIP
GET    /api/training/plates/export/preview  count without downloading
GET    /api/training/plates/:id
PUT    /api/training/plates/:id           update status / corrected_text / char_labels
POST   /api/training/plates/bulk          bulk status update
POST   /api/training/plates/finetune        start fine-tune job (dispatches to AI worker)
GET    /api/training/plates/finetune/status finetune job status + log
GET    /api/training/models/versions         list trained versions (+ active)
POST   /api/training/models/versions/:v/deploy  activate a version
GET    /api/models/manifest                 S3 model files (worker reads on startup)
GET    /api/models/download/:filename       stream model file from S3
PUT    /api/models/upload/:version/:filename worker uploads trained model back
POST   /api/models/push                     manually push local models → S3
```

---

## Fine-tuning Flow (training runs ON the AI worker — it has the GPU)

**Key principle:** the Go control plane does NOT run Python/training. It exports
data and dispatches; the **AI worker** trains in-process and uploads the result back.

```
Frontend → POST /api/training/plates/finetune
  Go (startFinetune, training_handlers.go):
    1. exportDatasetToDir()  — approved samples → YOLO dirs (train/valid) + data.yaml
    2. zipDirectory()        — zip the export
    3. S3 PutObject          — upload finetune_dataset_<ts>.zip to SnapshotsBucket
    4. BroadcastFinetuneTask — gRPC FrameTask{start_finetune, dataset_s3_key, epochs}
       (sendBlocking, 10s; aborts + deletes S3 zip if no worker connected)

  AI worker (_run_finetune, ai_worker_grpc.py) — runs in ThreadPoolExecutor:
    1. download zip via http://<cp>:8000/api/static/snapshots/<key>
    2. _rewrite_dataset_yaml() — rewrite data.yaml paths to THIS machine's extract dir
       (control plane's absolute paths are invalid on the worker)
    3. finetune_char_model.run_finetune_inline() — IN-PROCESS (no subprocess; works
       in PyInstaller frozen exe). Merges Roboflow datasets from ROBOFLOW_DATASET_BASE
       (data/datasets/, sibling of data/models/) if present, else CCTV-only.
    4. on "done": _upload_model_file() PUTs best .pt/.onnx → /api/models/upload/<ver>/...
    5. streams FinetuneProgress back via gRPC InferenceResult.finetune_progress

  Go (handleInferenceResult → FinetuneProgress, grpc_server.go):
    - epoch/info → update finetuneJob state + log
    - done → writeVersionMeta() + activateVersion()
             (copy versions/<ver>/* → active model dir, push S3, BroadcastReloadModels)
    - error → finetuneJob.setError()
```

### Critical invariants
- **Trained model lives on the worker** — it MUST upload back to the control plane
  (`PUT /api/models/upload`); `pushModelsToS3` reads the control plane's local disk.
- **Paths are absolute** — `_resolve_models_dir()` returns `os.path.abspath(...)`
  because ultralytics changes CWD during training (relative paths break post-train).
- **PyInstaller frozen exe**: `ai_worker_gui.py` calls `multiprocessing.freeze_support()`;
  training uses `workers=0` when `sys.frozen` (dataloader subprocs would relaunch the exe).
  `finetune_char_model` is bundled via `--hidden-import` (no `__file__`/subprocess).
- **DirectML build**: `build_win.bat directml` adds `--collect-binaries torch_directml`.

### Roboflow datasets (optional, accuracy boost)
Place on the **worker** at `data/datasets/` (sibling of `data/models/`):
`Thai-License-Plate-Character-Recognition-10`, `Thai-LNPR-3`, `LRU-License-Plate-1`,
`license-plate-charecter-5`. All remapped to MASTER_CLASSES before merge.
Missing dirs are skipped → CCTV-only training.

### char_labels JSON format
```json
[{"class_name": "ก", "cx": 0.12, "cy": 0.5, "bw": 0.08, "bh": 0.9, "confidence": 0.82}]
```
Coordinates are YOLO-normalized (0–1) relative to the deskewed plate crop.

### MASTER_CLASSES (`masterClasses` in `training_handlers.go`)
129 classes: digits `0–9`, Thai chars `A01–A44` (code-mapped), province codes `ACR…YST`.
Must match `MASTER_CLASSES` in `test-license-plate/train_char_model.py`.

---

## DB Models (GORM, `go-control-plane/db.go`)
- `Person`, `PersonFace` — face recognition subjects
- `Camera` — camera config with `detect_mode: face | plate | both`
- `DetectionLog` — face detection events
- `PlateDetectionLog` — plate detection events
- `PlateTrainingSample` — low-confidence crops for retraining

AutoMigrate runs all models on startup.

---

## Frontend (`frontend/src/`)
- `api/client.ts` — all API types + `trainingApi` object
- `pages/TrainingReview.tsx` — training review grid with canvas bbox overlay
- Route: `/training`
- Sidebar nav entry: 🎓 Training Review

---

## Go Control Plane Notes
- `go-control-plane/plate_handlers.go` — saves plate detection to S3 with loose crop (+1.5×w, +3×h padding)
- `go-control-plane/grpc_server.go` — processes gRPC results; calls `handlePlateDetections` + `saveTrainingFrames`
- S3 static serving: `GET /api/static/snapshots/:filename` → `GetSnapshotFromS3`
- `AfterFind` on `PlateTrainingSample` rewrites `image_path` (S3 key) → `/api/static/snapshots/{key}` URL

---

## Common Commands
```bash
# Build Go
cd go-control-plane && go build ./...

# TypeScript check
cd frontend && npx tsc --noEmit

# Python syntax check
python3 -c "import ast; ast.parse(open('backend/ai_worker_grpc.py').read())"
python3 -c "import ast; ast.parse(open('backend/finetune_char_model.py').read())"

# Run Python worker
cd backend && python3 ai_worker_grpc.py

# Build Windows worker exe (cpu | gpu | directml | openvino)
cd backend && build_win.bat directml --clean
```
