import logging
import os
import queue
import time
import sys
import threading
import uuid

import cv2
import grpc
import numpy as np

# Add current directory to path to ensure protobuf imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import facerec_pb2
import facerec_pb2_grpc
from app.face_engine import face_engine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("AI_Worker_gRPC")


class FaceTrack:
    """Represents a tracked face on a camera over a short buffering window."""

    def __init__(self, camera_id, bbox, embedding, task_id, image_bytes):
        self.camera_id = camera_id
        self.track_id = str(uuid.uuid4())
        self.bbox = bbox
        self.embedding = embedding
        self.task_id = task_id
        self.image_bytes = image_bytes
        self.face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        self.first_seen = time.time()
        self.last_seen = time.time()

    def update(self, bbox, embedding, task_id, image_bytes):
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area > self.face_area:
            self.bbox = bbox
            self.embedding = embedding
            self.task_id = task_id
            self.image_bytes = image_bytes
            self.face_area = area
        self.last_seen = time.time()


# Active tracks and cooldown state
active_tracks = {}
tracks_lock = threading.Lock()

cooldowns = []
cooldowns_lock = threading.Lock()

TRACK_TIMEOUT = 3.0  # seconds of inactivity before flushing
TRACK_MAX_DURATION = 5.0  # max seconds a track can run before flushing
COOLDOWN_DURATION = 30.0  # seconds


def clean_cooldowns():
    global cooldowns
    now = time.time()
    with cooldowns_lock:
        cooldowns = [c for c in cooldowns if c["expires_at"] > now]


def is_on_cooldown(embedding):
    clean_cooldowns()
    with cooldowns_lock:
        for c in cooldowns:
            # Cosine similarity (dot product of normalized embeddings)
            sim = np.dot(embedding, c["embedding"])
            if sim > 0.6:
                return True
    return False


def add_cooldown(embedding):
    with cooldowns_lock:
        cooldowns.append({
            "embedding": embedding,
            "expires_at": time.time() + COOLDOWN_DURATION
        })


def track_flusher(send_queue, stop_event=None):
    logger.info("Background Face Track Flusher thread started.")
    while True:
        if stop_event and stop_event.is_set():
            logger.info("Flusher stopping due to stop event.")
            break
        try:
            time.sleep(0.5)
            now = time.time()
            to_flush = []
            
            with tracks_lock:
                keys = list(active_tracks.keys())
                for key in keys:
                    track = active_tracks[key]
                    if (now - track.last_seen > TRACK_TIMEOUT) or (now - track.first_seen > TRACK_MAX_DURATION):
                        to_flush.append(track)
                        del active_tracks[key]
                        
            for track in to_flush:
                logger.info(f"Flushing best face track for camera {track.camera_id} (Face Area: {track.face_area})")
                add_cooldown(track.embedding)
                
                result = facerec_pb2.InferenceResult(
                    task_id=track.task_id,
                    detections=[facerec_pb2.Detection(
                        bbox=track.bbox,
                        embedding=track.embedding.tolist()
                    )]
                )
                send_queue.put(result)
        except Exception as e:
            logger.error(f"Error in track_flusher: {e}")


def run_grpc_client(control_plane_url=None, onnx_provider=None, stop_event=None):
    if control_plane_url is None:
        control_plane_url = os.getenv("CONTROL_PLANE_URL", "localhost:50051")
        
    if onnx_provider:
        os.environ["ONNX_PROVIDER"] = onnx_provider
    
    # Initialize InsightFace model
    logger.info("Initializing Face Engine...")
    face_engine.initialize()
    logger.info("Face Engine initialized successfully.")

    def sleep_interruptible(seconds):
        steps = int(seconds / 0.2)
        for _ in range(steps):
            if stop_event and stop_event.is_set():
                return
            time.sleep(0.2)

    while True:
        if stop_event and stop_event.is_set():
            logger.info("Stop event detected in main loop. Exiting client.")
            break

        logger.info(f"Connecting to gRPC server at {control_plane_url}...")
        try:
            # Configure gRPC channel options for large messages (up to 20MB)
            options = [
                ('grpc.max_receive_message_length', 20 * 1024 * 1024),
                ('grpc.max_send_message_length', 20 * 1024 * 1024),
            ]
            
            with grpc.insecure_channel(control_plane_url, options=options) as channel:
                stub = facerec_pb2_grpc.FaceInferenceServiceStub(channel)
                
                # Thread-safe queue for sending inference results back to server
                send_queue = queue.Queue()
                
                # Start track flusher thread
                flusher_thread = threading.Thread(target=track_flusher, args=(send_queue, stop_event), daemon=True)
                flusher_thread.start()
                
                def request_generator():
                    while True:
                        if stop_event and stop_event.is_set():
                            send_queue.put(None)
                            break
                        try:
                            item = send_queue.get(timeout=0.5)
                            if item is None:
                                break
                            yield item
                        except queue.Empty:
                            continue

                # Start the bidirectional stream
                responses = stub.ProcessStream(request_generator())
                logger.info("gRPC stream established. Listening for tasks...")
                
                for task in responses:
                    if stop_event and stop_event.is_set():
                        break

                    task_id = task.task_id
                    image_data = task.image_data
                    is_reg = task.is_registration
                    
                    logger.debug(f"Received task: id={task_id}, size={len(image_data)} bytes, is_registration={is_reg}")
                    
                    try:
                        # Decode image from JPEG bytes
                        nparr = np.frombuffer(image_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if img is None:
                            logger.error(f"Failed to decode JPEG image for task {task_id}")
                            if is_reg:
                                send_queue.put(facerec_pb2.InferenceResult(task_id=task_id, detections=[]))
                            continue
                            
                        if is_reg:
                            # Registration mode: extract embedding and return immediately
                            emb = face_engine.extract_embedding_from_image(img)
                            if emb is not None:
                                result = facerec_pb2.InferenceResult(
                                    task_id=task_id,
                                    detections=[facerec_pb2.Detection(
                                        bbox=[0.0, 0.0, 0.0, 0.0],
                                        embedding=emb.tolist()
                                    )]
                                )
                                send_queue.put(result)
                            else:
                                logger.warning(f"No face detected in registration image for task {task_id}")
                                send_queue.put(facerec_pb2.InferenceResult(task_id=task_id, detections=[]))
                        else:
                            # Parse camera_id from task_id (cameraID_timestamp_uuid)
                            parts = task_id.split('_')
                            camera_id = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
                            
                            faces = face_engine.detect_faces(img)
                            
                            with tracks_lock:
                                for face in faces:
                                    emb = face.embedding
                                    bbox = face.bbox
                                    
                                    # Check if on cooldown
                                    if is_on_cooldown(emb):
                                        continue
                                        
                                    # Try to associate with active tracks for this camera
                                    matched_track = None
                                    for track in active_tracks.values():
                                        if track.camera_id == camera_id:
                                            sim = np.dot(emb, track.embedding)
                                            if sim > 0.6:
                                                matched_track = track
                                                break
                                                
                                    if matched_track:
                                        matched_track.update(bbox, emb, task_id, image_data)
                                    else:
                                        # Create new track
                                        new_track = FaceTrack(camera_id, bbox, emb, task_id, image_data)
                                        active_tracks[new_track.track_id] = new_track
                                        
                    except Exception as ex:
                        logger.error(f"Error processing task {task_id}: {ex}")
                        if is_reg:
                            send_queue.put(facerec_pb2.InferenceResult(task_id=task_id, detections=[]))
                            
        except grpc.RpcError as e:
            logger.error(f"gRPC stream connection error: {e.details() if hasattr(e, 'details') else e}. Retrying in 5 seconds...")
            sleep_interruptible(5)
        except Exception as e:
            logger.error(f"Unexpected error: {e}. Retrying in 5 seconds...")
            sleep_interruptible(5)


if __name__ == "__main__":
    run_grpc_client()
