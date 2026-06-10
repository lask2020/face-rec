FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required by OpenCV and InsightFace
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Compile protobuf
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. facerec.proto

# Expose API port
EXPOSE 8899

# Start the gRPC client
CMD ["python", "ai_worker_grpc.py"]
