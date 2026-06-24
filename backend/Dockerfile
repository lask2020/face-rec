# syntax=docker/dockerfile:1
# Enable BuildKit cache mounts for apt and pip — dramatically speeds up rebuilds.
#
# GPU_TYPE build arg selects the GPU backend automatically:
#   cpu    (default) — onnxruntime CPU, torch CPU
#   cuda              — onnxruntime-gpu, torch CUDA 12.1
#   rocm              — onnxruntime-rocm, torch ROCm 6.2  (AMD GPU on Linux)
#
# Usage:
#   docker build --build-arg GPU_TYPE=rocm -t ai-worker .
#   or via docker-compose (see GPU_TYPE arg in compose file)

ARG GPU_TYPE=cpu

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.9-slim AS builder

ARG GPU_TYPE

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        build-essential

COPY requirements.txt .

# Install base deps (includes CPU onnxruntime from requirements.txt)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Override torch + onnxruntime for GPU builds
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$GPU_TYPE" = "cuda" ]; then \
        pip install --upgrade \
            torch torchvision \
            --index-url https://download.pytorch.org/whl/cu121 && \
        pip install --upgrade onnxruntime-gpu; \
    elif [ "$GPU_TYPE" = "rocm" ]; then \
        pip install --upgrade \
            torch torchvision \
            --index-url https://download.pytorch.org/whl/rocm6.2 && \
        pip install --upgrade onnxruntime-rocm; \
    fi


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
# Lean image: no build tools, no wget, no model files baked in.
# Models (CodeFormer, Real-ESRGAN) are downloaded automatically on first start
# and persisted via the /app/data volume.
FROM python:3.9-slim AS runtime

WORKDIR /app

# Only the runtime .so deps — no build-essential
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code (fast — .dockerignore keeps context small)
COPY . .

# Compile protobuf (no network, pure local)
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. facerec.proto

# Models are downloaded at runtime by face_restorer.py / face_super_resolver.py.
# Mount a named volume here to persist them across container restarts:
#   docker run -v face_models:/app/data ...
#   or via docker-compose: volumes: - face_models:/app/data
VOLUME /app/data

EXPOSE 8899
CMD ["python", "ai_worker_grpc.py"]
