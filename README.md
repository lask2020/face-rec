# Face Recognition CCTV System — Distributed Architecture

ระบบวิเคราะห์และจดจำใบหน้าจากกล้อง CCTV แบบกระจายศูนย์ (Distributed Microservices) รองรับกล้องหลายตัวพร้อมกันแบบ Real-time โดยใช้ Redis Stream เป็นตัวกลางส่งเฟรม, Python AI Worker ประมวลผล InsightFace ผ่าน gRPC Bidirectional Stream, และ React Dashboard สำหรับดูสดและบริหารจัดการระบบ

---

## สถาปัตยกรรมระบบ (System Architecture)

```
CCTV Cameras
    │ RTSP Stream
    ▼
Go2RTC Stream Hub  ──────────────────────────────────────┐
    │ HTTP JPEG Frames                                    │ WebRTC (Frontend)
    ▼                                                     │
Go Ingestion Worker                                       │
    │ Publish JPEG + camera_id + ts                       │
    ▼                                                     │
Redis Stream (image.queue)                                │
    │ XReadGroup                                          │
    ▼                                                     │
Go Control Plane  ◄──────────────────────────────────────┘
    │ gRPC Bidirectional Stream (FrameTask / InferenceResult)
    ▼
Python AI Worker(s) [Stateless Fleet]
    │ InsightFace buffalo_l + ArcFace 512-dim Embedding
    │ Face Tracking + Quality Buffer + Cooldown
    │ CodeFormer Face Restoration (optional)
    └────────────────────────────────────────────────────►
                                                Go Control Plane
                                                    │
                                         ┌──────────┼──────────────┐
                                         ▼          ▼              ▼
                                     Qdrant      RustFS S3     PostgreSQL
                                  (Vector DB)  (Snapshots,   (Persons, Cameras,
                                                Crops, etc.)  Detection Logs)
                                                    │
                                                    ▼
                                         React Admin Dashboard
                                       (REST API + WebSocket)
```

---

## รายละเอียดแต่ละ Component

### 1. Go2RTC Stream Hub (Port 1984 / 8554 / 8555)
- รับ RTSP Stream จากกล้อง CCTV
- แปลงและกระจาย Stream ไปยัง Ingestion Worker (JPEG) และ Frontend (WebRTC)
- ตั้งค่าใน [`go2rtc.yaml`](go2rtc.yaml)

### 2. Go Ingestion Worker (`go-ingestion`)
- ดึงเฟรม JPEG จาก go2rtc ผ่าน HTTP Snapshot
- ส่ง `camera_id`, `ts` (Unix ms), และ `data` (JPEG bytes) เข้า Redis Stream `image.queue`
- รองรับ FPS throttle ต่อกล้อง

### 3. Go Control Plane (`go-control-plane`) — Port 8000 (REST) / 50051 (gRPC)
ส่วนหลักของระบบ มีหน้าที่หลัก 4 อย่าง:

| หน้าที่ | รายละเอียด |
|---------|-----------|
| **Frame Dispatcher** | อ่านเฟรมจาก Redis Stream และส่งไปยัง AI Worker ที่ว่าง (Sticky Routing ต่อกล้อง + Rebalance อัตโนมัติ) |
| **gRPC Server** | รับ `InferenceResult` จาก Worker, ค้นหา embedding ใน Qdrant, บันทึก Log/Snapshot/Crop ลง PostgreSQL + S3 |
| **REST API** | CRUD Cameras, Persons (Face Registration), Detection Logs + Static file proxy |
| **WebSocket Broadcast** | Push real-time detection events ไปยัง Frontend ทุก Client พร้อมกัน |

**Sticky Camera Routing**: แต่ละกล้องจะถูก Assign ให้ Worker คนเดิมตลอด เพื่อให้ Face Tracking บน Worker ทำงานได้ถูกต้อง เมื่อมี Worker ใหม่เชื่อมต่อเข้ามาจะทำ Rebalance อัตโนมัติ

### 4. Python AI Worker (`backend`) — Stateless gRPC Client

**Pipeline การประมวลผล:**

```
รับ FrameTask (JPEG bytes, task_id, is_registration)
    │
    ├─ [Registration Mode] extract_embedding_from_image()
    │      ├─ Single-face gate
    │      ├─ Frontality gate (±15°)
    │      ├─ Sharpness gate (Laplacian ≥ 60)
    │      └─ ส่ง embedding กลับทันที
    │
    └─ [Detection Mode] detect_faces()
           ├─ InsightFace buffalo_l (SCRFD Detection + ArcFace 512-dim)
           ├─ Filter: det_score ≥ 0.5, face_size ≥ 45px, frontality ≤ 20°
           └─ Face Tracking + Quality Buffer (per camera_id)
                  │
                  ├─ Match ด้วย Cosine Similarity > 0.6
                  ├─ Quality Score = face_area × frontality
                  ├─ เลือกเฟรมที่ดีที่สุด (sharp > blurry → quality score)
                  └─ Flush เมื่อ timeout (3s) หรือ max_duration (5s)
                         │
                         ├─ Cooldown check (Cosine > 0.6 → skip 30s)
                         ├─ Sharpness check (MIN_TRACK_SHARPNESS ≥ 30)
                         ├─ [Optional] CodeFormer Face Restoration
                         └─ ส่ง InferenceResult กลับ Control Plane
```

**Hardware Acceleration** — รองรับ ONNX Runtime Execution Providers:
- `CoreMLExecutionProvider` — Apple Silicon / macOS
- `CUDAExecutionProvider` — NVIDIA GPU
- `ROCmExecutionProvider` — AMD GPU (Linux)
- `OpenVINOExecutionProvider` — Intel CPU/GPU
- `DmlExecutionProvider` — Windows DirectML (AMD/Intel/NVIDIA)
- `CPUExecutionProvider` — CPU Fallback

**GPU Compatibility Fixes ที่ทำไว้:**
- Patch SCRFD output tensor ordering สำหรับ DirectML/non-CPU providers
- Patch `FaceAnalysis.get()` เพื่อ handle `kps=None` และ Thread-safety lock สำหรับ GPU inference
- Pose estimation fallback จาก 5 facial landmarks เมื่อไม่มี `pose` attribute

### 5. React Admin Dashboard (`frontend`) — Port 80

| หน้า | ฟีเจอร์ |
|------|---------|
| **Live Monitor** | ดูกล้องสดผ่าน WebRTC (go2rtc) + Real-time detection alerts ผ่าน WebSocket |
| **Camera Management** | CRUD กล้อง (ชื่อ, RTSP URL, FPS Process), Start/Stop stream |
| **Person Management** | ลงทะเบียนใบหน้า (อัปโหลดหลายรูป, Quality gate), แก้ไข/ลบ |
| **Detection Logs** | ประวัติการตรวจจับ — Snapshot, Face Crop, Restored Face (CodeFormer), เปรียบเทียบ Side-by-Side |
| **AI Workers** | ดูสถานะ Worker ที่เชื่อมต่อ, Avg process time, Pause/Resume |
| **Signage Dashboard** | หน้าจอแสดงผลแยกสำหรับติดจอสาธารณะ (Real-time alert pop-up) |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Control Plane | Go 1.20+, Fiber v2, GORM, gRPC Server |
| Ingestion Worker | Go 1.20+ |
| AI Worker | Python 3.10+, InsightFace (buffalo_l), ONNX Runtime, OpenCV, gRPC |
| Face Restoration | CodeFormer (ONNX) — optional |
| Vector Database | Qdrant (512-dim cosine similarity search) |
| Relational Database | PostgreSQL 15 (Persons, Cameras, Detection Logs) |
| Object Storage | RustFS (S3-compatible — Snapshots, Face Crops, Restored Faces) |
| Message Broker | Redis 7 (Streams) |
| Media Stream | go2rtc (RTSP → WebRTC/HLS/JPEG) |
| Frontend | React 18, TypeScript, Vite, WebSocket |
| Containerization | Docker + Docker Compose |

---

## Quick Start (Docker Compose)

### Prerequisites
- Docker + Docker Compose (หรือ Docker Desktop)

### ขั้นตอน

**1. ตั้งค่ากล้อง CCTV ใน `go2rtc.yaml`:**
```yaml
streams:
  cam_1:
    - rtsp://admin:password@192.168.1.100/stream1
  cam_2:
    - rtsp://admin:password@192.168.1.101/stream1
```

**2. เริ่มต้นระบบด้วย `deploy.sh`:**
```bash
chmod +x deploy.sh

# Full redeploy (down → build → up)
./deploy.sh

# Rebuild เฉพาะ service
./deploy.sh ai-worker
./deploy.sh control-plane
./deploy.sh ingestion
./deploy.sh frontend

# ดู status และ logs
./deploy.sh status
./deploy.sh logs
./deploy.sh logs ai-worker
```

หรือรันตรงด้วย Docker Compose:
```bash
docker compose up --build -d
```

> **หมายเหตุ:** `ai-worker` ใน `docker-compose.yml` ถูก comment ไว้โดย default  
> เพราะ AI Worker มักรันแยกบน Windows/macOS เพื่อใช้ GPU ของเครื่อง  
> (ดูส่วน Windows EXE / Native Run ด้านล่าง)

### Port Map

| URL | Service |
|-----|---------|
| http://localhost | Frontend Dashboard |
| http://localhost:8000 | Control Plane REST API |
| http://localhost:1984 | go2rtc Web UI |
| http://localhost:9001 | RustFS Console (admin / admin12345) |
| http://localhost:6333/dashboard | Qdrant Dashboard |

---

## โครงสร้างโปรเจกต์

```
face-rec/
├── backend/                        # Python AI Worker
│   ├── app/
│   │   ├── face_engine.py          # InsightFace wrapper, FaceResult, quality scoring
│   │   ├── face_restorer.py        # CodeFormer ONNX face restoration (optional)
│   │   └── gpu_lock.py             # Global inference lock for GPU thread safety
│   ├── ai_worker_grpc.py           # gRPC client, FaceTrack, quality buffer, cooldown
│   ├── ai_worker_gui.py            # Windows GUI wrapper (PyQt6)
│   ├── data/
│   │   └── codeformer.onnx         # CodeFormer model (optional)
│   ├── facerec_pb2.py              # Generated protobuf (Python)
│   ├── facerec_pb2_grpc.py         # Generated gRPC stubs (Python)
│   ├── requirements.txt
│   ├── build_win.bat               # Windows PyInstaller build script
│   ├── build_mac.sh                # macOS PyInstaller build script
│   └── Dockerfile
├── go-control-plane/               # Go API & Controller
│   ├── main.go                     # Entry point, Fiber router, WebSocket hub
│   ├── grpc_server.go              # gRPC server, frame dispatcher, sticky routing
│   ├── handlers.go                 # REST handlers (Cameras, Persons, Detections, Workers)
│   ├── db.go                       # GORM models (Camera, Person, FaceVector, DetectionLog)
│   ├── qdrant.go                   # Qdrant vector upsert/search
│   ├── s3.go                       # RustFS/S3 upload helper
│   ├── draw.go                     # BBox drawing, JPEG crop utilities
│   ├── redis.go                    # Redis client + camera assignment pub/sub
│   ├── synology.go                 # (Optional) Synology NAS integration
│   ├── facerec/                    # Generated protobuf (Go)
│   └── Dockerfile
├── go-ingestion/                   # Go Frame Ingestion Worker
│   ├── main.go                     # go2rtc snapshot poll → Redis Stream push
│   └── Dockerfile
├── frontend/                       # React Admin Dashboard
│   ├── src/
│   │   ├── App.tsx                 # Router + layout
│   │   └── ...                     # Pages: Dashboard, Cameras, Persons, Detections, Signage
│   └── Dockerfile
├── facerec.proto                   # gRPC Protobuf definition (source of truth)
├── go2rtc.yaml                     # Media stream config
├── docker-compose.yml              # Service orchestration
├── deploy.sh                       # Deploy helper script
└── worker_config.json              # Native worker config (URL + ONNX provider)
```

---

## Configuration

### Environment Variables — AI Worker (`backend`)

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTROL_PLANE_URL` | `localhost:50051` | gRPC server address |
| `ONNX_PROVIDER` | (auto-detect) | Force a specific ONNX provider |
| `FACE_DETECTION_SIZE` | `640` | SCRFD detection input size (640 แนะนำสำหรับ CoreML) |
| `MIN_FACE_SIZE` | `45.0` | ขนาดใบหน้าขั้นต่ำ (px) ที่จะประมวลผล |
| `FACE_SHARPNESS_THRESHOLD` | `50.0` | Laplacian variance threshold สำหรับ tracking |
| `MIN_TRACK_SHARPNESS` | `30.0` | Sharpness ขั้นต่ำก่อน flush track |
| `AI_WORKER_CONCURRENCY` | `4` | จำนวน thread สำหรับประมวลผลเฟรมพร้อมกัน |
| `FACE_DATA_ROOT` | (auto) | โฟลเดอร์ InsightFace models |

**ค่าคงที่ใน `ai_worker_grpc.py`:**
- `TRACK_TIMEOUT = 3.0s` — หากไม่เห็นใบหน้าเกิน 3 วินาที → flush
- `TRACK_MAX_DURATION = 5.0s` — บังคับ flush หาก track นานเกิน 5 วินาที
- `COOLDOWN_DURATION = 30.0s` — cooldown หลัง flush เพื่อป้องกันการยิงซ้ำ

### Environment Variables — Go Control Plane

| Variable | Example |
|----------|---------|
| `REDIS_URL` | `redis:6379` |
| `POSTGRES_HOST` | `postgres` |
| `POSTGRES_PORT` | `5432` |
| `POSTGRES_USER` | `root` |
| `POSTGRES_PASSWORD` | `password` |
| `POSTGRES_DB` | `facerec` |
| `QDRANT_URL` | `http://qdrant:6333` |
| `S3_ENDPOINT` | `rustfs:9000` |
| `S3_ACCESS_KEY` | `admin` |
| `S3_SECRET_KEY` | `admin12345` |
| `S3_FACES_BUCKET` | `faces` |
| `S3_SNAPSHOTS_BUCKET` | `snapshots` |
| `GO2RTC_URL` | `http://go2rtc:1984` |

---

## Hardware Acceleration

### Docker (Linux)

เปิดไฟล์ [`backend/requirements.txt`](backend/requirements.txt) และเลือก ONNX Runtime package ที่ตรงกับฮาร์ดแวร์ (เปิดใช้แค่ตัวเดียว):

```txt
# CPU / Apple Silicon (default)
onnxruntime

# NVIDIA GPU
# onnxruntime-gpu

# Intel CPU/GPU (OpenVINO)
# onnxruntime-openvino

# AMD GPU (Linux ROCm)
# onnxruntime-rocm

# Windows DirectML (AMD/Intel/NVIDIA)
# onnxruntime-directml
```

จากนั้นแก้ไข `docker-compose.yml` ส่วน `ai-worker`:

```yaml
ai-worker:
  environment:
    - CONTROL_PLANE_URL=control-plane:50051
    - ONNX_PROVIDER=OpenVINOExecutionProvider   # Intel GPU
    # - ONNX_PROVIDER=CUDAExecutionProvider     # NVIDIA GPU
  # Passthrough GPU devices (Linux):
  # devices:
  #   - /dev/dri:/dev/dri
```

---

## Windows / macOS — Native AI Worker

AI Worker รองรับการรันนอก Docker เพื่อใช้ GPU ของเครื่องโดยตรง

### รัน Python โดยตรง

```bash
cd backend
pip install -r requirements.txt
python ai_worker_gui.py     # GUI mode (PyQt6)
# หรือ
python ai_worker_grpc.py    # CLI mode
```

### Build ไฟล์ EXE (Windows)

```bat
cd backend
build_win.bat
```
ไฟล์ EXE จะอยู่ที่ `backend/dist/FaceRec_AI_Worker.exe`

### Build ไฟล์ Binary (macOS)

```bash
cd backend
./build_mac.sh
```
ไฟล์จะอยู่ที่ `backend/dist/FaceRec_AI_Worker_macOS_CPU`

### GUI Usage

เมื่อเปิด GUI จะมีช่องให้กรอก:
- **Control Plane URL**: เช่น `192.168.1.100:50051`
- **Execution Provider**: เลือก provider ที่ตรงกับ GPU ของเครื่อง
- **Start/Stop Worker**: เริ่ม/หยุด Worker ได้ทันที
- **Log Window**: แสดงสถานะ Real-time ของทุกกล้อง

การตั้งค่าจะถูกบันทึกใน `worker_config.json` โดยอัตโนมัติ

### Native Run Script

```bash
# macOS/Linux
./run_native.sh

# Windows
run_native.bat
```

---

## gRPC Protocol

ดูไฟล์ [`facerec.proto`](facerec.proto) สำหรับ schema เต็ม

```protobuf
service FaceInferenceService {
  rpc ProcessStream (stream InferenceResult) returns (stream FrameTask);
}

message FrameTask {
  string task_id      = 1;  // "{camera_id}_{ts}_{uuid}" หรือ uuid สำหรับ registration
  bytes  image_data   = 2;  // JPEG bytes (สูงสุด 20MB)
  bool   is_registration = 3;
}

message InferenceResult {
  string task_id         = 1;
  repeated Detection detections = 2;
  string error_message   = 3;
  double process_time_ms = 4;  // ส่งทุก 2s เป็น metrics ("metrics" task_id)
}

message Detection {
  repeated float bbox              = 1;  // [x1, y1, x2, y2]
  repeated float embedding         = 2;  // 512-dim normalized ArcFace embedding
  bytes          restored_face_jpeg = 3; // CodeFormer restored face (optional)
}
```

---

## Face Quality Pipeline (สรุป)

```
เฟรมจากกล้อง
    ↓
SCRFD Detection (buffalo_l)
    ↓ filter: det_score ≥ 0.5, face ≥ 45px, frontality ≤ 20°
ArcFace Embedding (512-dim, L2-normalized)
    ↓
Worker Face Tracking (Cosine Sim > 0.6 → same person)
    ↓ quality score = face_area × frontality
Best Frame Selection (sharp beats blurry → higher quality wins)
    ↓ timeout 3s / max 5s
Cooldown Check (Cosine > 0.6 → skip 30s)
    ↓ sharpness ≥ 30
[Optional] CodeFormer Face Restoration
    ↓
ส่ง InferenceResult → Control Plane
    ↓
Qdrant Vector Search (similarity ≥ 0.4 → match Person)
    ↓
PostgreSQL Log + S3 Snapshot/Crop + WebSocket Broadcast
```

---

## API Reference (ย่อ)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/cameras` | รายการกล้องทั้งหมด |
| POST | `/api/cameras` | เพิ่มกล้อง |
| PUT | `/api/cameras/:id` | แก้ไขกล้อง |
| DELETE | `/api/cameras/:id` | ลบกล้อง |
| POST | `/api/cameras/:id/start` | เริ่ม stream |
| POST | `/api/cameras/:id/stop` | หยุด stream |
| GET | `/api/persons` | รายการบุคคลที่ลงทะเบียน |
| POST | `/api/persons` | เพิ่มบุคคลพร้อมรูปใบหน้า |
| PUT | `/api/persons/:id` | แก้ไขข้อมูล/รูปใบหน้า |
| DELETE | `/api/persons/:id` | ลบบุคคล |
| GET | `/api/detections` | ประวัติการตรวจจับ (paginated) |
| GET | `/api/workers` | รายการ AI Worker ที่เชื่อมต่ออยู่ |
| POST | `/api/workers/:id/pause` | Pause/Resume Worker |
| WS | `/ws` | WebSocket real-time detection events |
| GET | `/api/static/snapshots/:file` | Proxy รูปจาก S3 |
| GET | `/api/static/faces/:file` | Proxy รูปใบหน้าจาก S3 |
