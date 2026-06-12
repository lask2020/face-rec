import logging
import os
import queue
import time
import sys
import threading
import uuid
import warnings

# Suppress numpy FutureWarning from insightface
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

import cv2
import grpc
import numpy as np

# Add current directory to path to ensure protobuf imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import facerec_pb2
import facerec_pb2_grpc
from app.face_engine import face_engine, compute_sharpness, SHARPNESS_THRESHOLD
from app.face_restorer import face_restorer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("AI_Worker_gRPC")


class FaceTrack:
    """Represents a tracked face on a camera over a short buffering window."""

    def __init__(self, camera_id, bbox, embedding, task_id, image_bytes, sharpness: float = 0.0, frontality: float = 1.0, kps: list = None):
        self.camera_id = camera_id
        self.track_id = str(uuid.uuid4())
        self.bbox = bbox
        self.embedding = embedding
        self.task_id = task_id
        self.image_bytes = image_bytes
        self.face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        self.sharpness = sharpness
        self.frontality = frontality
        self.kps = kps
        self.quality_score = self.face_area * self.frontality
        
        self.first_seen = time.time()
        self.last_seen = time.time()

    def update(self, bbox, embedding, task_id, image_bytes, sharpness: float = 0.0, frontality: float = 1.0, kps: list = None):
        """Update the track's best frame using sharpness-gated + quality score selection.

        Quality score = face_area × frontality.
        This favors frames where the face is both large AND looking straight.

        Strategy:
        1. If the new frame is sharp and the current best is NOT sharp,
           always replace (sharp beats blurry regardless of quality).
        2. If both are sharp (or both are blurry), pick the one with
           the higher quality score (area × frontality).
        3. If the current best is sharp and the new frame is NOT,
           keep the current best (don't replace sharp with blurry).
        """
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        new_quality = area * frontality
        new_is_sharp = sharpness >= SHARPNESS_THRESHOLD
        cur_is_sharp = self.sharpness >= SHARPNESS_THRESHOLD

        should_replace = False
        if new_is_sharp and not cur_is_sharp:
            # New frame is sharp, current is blurry -> always replace
            should_replace = True
        elif new_is_sharp == cur_is_sharp:
            # Both sharp or both blurry -> pick higher quality score
            should_replace = new_quality > self.quality_score
        # else: current is sharp, new is blurry -> keep current

        if should_replace:
            self.bbox = bbox
            self.embedding = embedding
            self.task_id = task_id
            self.image_bytes = image_bytes
            self.face_area = area
            self.sharpness = sharpness
            self.frontality = frontality
            self.kps = kps
            self.quality_score = new_quality
        self.last_seen = time.time()


# Active tracks and cooldown state
active_tracks = {}
tracks_lock = threading.Lock()

cooldowns = []
cooldowns_lock = threading.Lock()

# Camera state and logs suppression
last_seen_camera = {}
cameras_lock = threading.Lock()

# Stats for average process time tracking
stats_lock = threading.Lock()
recent_process_times = []

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
    last_stats_sent = 0
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
                # Discard track if the best frame is still extremely blurry
                min_track_sharpness = float(os.getenv("MIN_TRACK_SHARPNESS", "30.0"))
                if track.sharpness < min_track_sharpness:
                    logger.info(f"Discarding track {track.track_id} on camera {track.camera_id} because best frame sharpness ({track.sharpness:.1f}) is below minimum threshold ({min_track_sharpness:.1f})")
                    continue

                logger.info(f"Flushing best face track for camera {track.camera_id} (Area: {track.face_area:.0f}, Sharpness: {track.sharpness:.1f}, Frontality: {track.frontality:.2f}, Quality: {track.quality_score:.0f})")
                add_cooldown(track.embedding)
                
                restored_face_bytes = b""
                if face_restorer.is_enabled():
                    try:
                        nparr = np.frombuffer(track.image_bytes, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is not None:
                            if getattr(track, 'kps', None) is not None:
                                from insightface.utils import face_align
                                face_crop = face_align.norm_crop(img, np.array(track.kps), image_size=512)
                            else:
                                h, w = img.shape[:2]
                                cx = int((track.bbox[0] + track.bbox[2]) / 2)
                                cy = int((track.bbox[1] + track.bbox[3]) / 2)
                                bw = int(track.bbox[2] - track.bbox[0])
                                bh = int(track.bbox[3] - track.bbox[1])
                                side = max(bw, bh)
                                padding = int(side * 0.5)

                                x1 = max(0, cx - padding)
                                y1 = max(0, cy - padding)
                                x2 = min(w, cx + padding)
                                y2 = min(h, cy + padding)
                                face_crop = img[y1:y2, x1:x2]

                            if face_crop is not None and face_crop.size > 0:
                                restored = face_restorer.restore_face(face_crop)
                                if restored is not None:
                                    success, encoded_img = cv2.imencode(".jpg", restored)
                                    if success:
                                        restored_face_bytes = encoded_img.tobytes()
                                        logger.info(f"Successfully restored face for track {track.track_id}")
                    except Exception as e:
                        logger.error(f"Error during face restoration in flusher: {e}")

                result = facerec_pb2.InferenceResult(
                    task_id=track.task_id,
                    detections=[facerec_pb2.Detection(
                        bbox=track.bbox,
                        embedding=track.embedding.tolist(),
                        restored_face_jpeg=restored_face_bytes
                    )]
                )
                send_queue.put(result)

            # Periodically send metrics update (every 2 seconds)
            if now - last_stats_sent > 2.0:
                last_stats_sent = now
                with stats_lock:
                    if recent_process_times:
                        avg_ms = sum(recent_process_times) / len(recent_process_times)
                        result = facerec_pb2.InferenceResult(
                            task_id="metrics",
                            detections=[],
                            process_time_ms=avg_ms
                        )
                        send_queue.put(result)

            # Clean up inactive cameras to log stop events
            inactive_timeout = 10.0
            with cameras_lock:
                for cam_id, last_ts in list(last_seen_camera.items()):
                    if now - last_ts > inactive_timeout:
                        logger.info(f"Stopped processing stream for Camera {cam_id} (inactive)")
                        del last_seen_camera[cam_id]
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
    
    # Initialize CodeFormer Face Restorer
    if face_restorer.is_enabled():
        logger.info("Initializing Face Restorer...")
        face_restorer.initialize()
        logger.info("Face Restorer initialized successfully.")

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
                        start_time = time.time()
                        # Decode image from JPEG bytes
                        nparr = np.frombuffer(image_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if img is None:
                            logger.error(f"Failed to decode JPEG image for task {task_id}")
                            if is_reg:
                                duration_ms = (time.time() - start_time) * 1000
                                send_queue.put(facerec_pb2.InferenceResult(task_id=task_id, detections=[], process_time_ms=duration_ms))
                            continue
                            
                        if is_reg:
                            # Registration mode: extract embedding and return immediately
                            emb, err_msg = face_engine.extract_embedding_from_image(img)
                            duration_ms = (time.time() - start_time) * 1000
                            if err_msg is not None:
                                logger.warning(f"Registration face rejected: {err_msg}")
                                result = facerec_pb2.InferenceResult(
                                    task_id=task_id,
                                    detections=[],
                                    error_message=err_msg,
                                    process_time_ms=duration_ms
                                )
                                send_queue.put(result)
                            elif emb is not None:
                                result = facerec_pb2.InferenceResult(
                                    task_id=task_id,
                                    detections=[facerec_pb2.Detection(
                                        bbox=[0.0, 0.0, 0.0, 0.0],
                                        embedding=emb.tolist()
                                    )],
                                    process_time_ms=duration_ms
                                )
                                send_queue.put(result)
                            else:
                                logger.warning(f"No face detected in registration image for task {task_id}")
                                send_queue.put(facerec_pb2.InferenceResult(task_id=task_id, detections=[], error_message="No face detected", process_time_ms=duration_ms))
                        else:
                            # Parse camera_id from task_id (cameraID_timestamp_uuid)
                            parts = task_id.split('_')
                            camera_id = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
                            
                            now_time = time.time()
                            with cameras_lock:
                                if camera_id not in last_seen_camera:
                                    logger.info(f"Started processing stream for Camera {camera_id}")
                                last_seen_camera[camera_id] = now_time
                            
                            faces = face_engine.detect_faces(img)
                            
                            duration_ms = (time.time() - start_time) * 1000
                            logger.info(f"Processed frame for Camera {camera_id} in {duration_ms:.1f}ms - detected {len(faces)} face(s)")
                            
                            with stats_lock:
                                recent_process_times.append(duration_ms)
                                if len(recent_process_times) > 50:
                                    recent_process_times.pop(0)
                            
                            with tracks_lock:
                                for face in faces:
                                    emb = face.embedding
                                    bbox = face.bbox
                                    sharpness = face.sharpness
                                    frontality = face.frontality
                                    
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
                                        matched_track.update(bbox, emb, task_id, image_data, sharpness, frontality, face.kps)
                                    else:
                                        # Create new track
                                        new_track = FaceTrack(camera_id, bbox, emb, task_id, image_data, sharpness, frontality, face.kps)
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
