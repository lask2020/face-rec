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
- `_assemble_multi_frame()` — majority-vote on char count, per-slot best confidence, province by majority
- **Fuzzy cooldown**: Levenshtein distance ≤ `PLATE_FUZZY_DIST` (default 2) prevents duplicate DB records from OCR-variant plates

### Key env vars (Python worker)
| Var | Default | Purpose |
|-----|---------|---------|
| `PLATE_FUZZY_DIST` | `2` | Max edit distance for duplicate suppression |
| `PLATE_TRACK_TIMEOUT` | `6.0` | Seconds before flushing idle track |
| `PLATE_TRACK_MAX_DURATION` | `12.0` | Max seconds to accumulate a track |
| `MIN_PLATE_FLUSH_CONF` | `0.25` | Discard tracks below this confidence |
| `MIN_PLATE_HITS` | `1` | Minimum frames required to flush |
| `TRAINING_CAPTURE_CONF_MAX` | `0.65` | Frames below this go to training dataset |

---

## Training / Active-Learning Pipeline

Low-confidence plate crops are automatically captured and saved for OCR retraining.

### Data flow
1. Python worker: frames with `confidence < TRAINING_CAPTURE_CONF_MAX` → `PlateTrainingFrame` in gRPC `InferenceResult`
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
```

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

# Run Python worker
cd backend && python3 ai_worker_grpc.py
```
