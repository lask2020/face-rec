# syntax=docker/dockerfile:1
# Enable BuildKit cache mounts for apt and pip — dramatically speeds up rebuilds.

# ── Stage 1: builder ─────────────────────────────────────────────────────────
# Has build-essential for compiling native wheels (insightface, opencv, etc.)
# Not included in the final image.
FROM python:3.9-slim AS builder

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        build-essential

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt


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
