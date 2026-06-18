import threading

# Global lock to serialize ALL ONNX Runtime session.run() calls across the process.
# This prevents DirectML Execution Provider from crashing or corrupting memory
# when multiple threads (e.g. detection workers and background face restorer)
# attempt to use the GPU simultaneously.
inference_lock = threading.Lock()
