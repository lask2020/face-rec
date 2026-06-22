package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"net"
	"sync"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
	"github.com/minio/minio-go/v7"
	"github.com/redis/go-redis/v9"
	"google.golang.org/grpc"

	facerec "github.com/face-rec/go-control-plane/facerec"
)

// detectModeCache caches camera detect_mode values so the dispatcher doesn't
// hit the DB on every frame. Entries are refreshed lazily after cacheTTL.
var (
	detectModeCache   = make(map[uint]detectModeCacheEntry)
	detectModeCacheMu sync.Mutex
	cacheTTL          = 8 * time.Second
)

type detectModeCacheEntry struct {
	mode      string
	expiresAt time.Time
}

func getCachedDetectMode(cameraID uint) string {
	detectModeCacheMu.Lock()
	defer detectModeCacheMu.Unlock()

	if entry, ok := detectModeCache[cameraID]; ok && time.Now().Before(entry.expiresAt) {
		return entry.mode
	}

	// Cache miss or expired — query DB and refresh
	var cam Camera
	mode := "face"
	if DB.Select("detect_mode").First(&cam, cameraID).Error == nil && cam.DetectMode != "" {
		mode = cam.DetectMode
	}
	detectModeCache[cameraID] = detectModeCacheEntry{mode: mode, expiresAt: time.Now().Add(cacheTTL)}
	return mode
}

// InvalidateDetectModeCache removes a single camera from the cache so the next
// frame picks up any detect_mode change immediately (call this after UpdateCamera).
func InvalidateDetectModeCache(cameraID uint) {
	detectModeCacheMu.Lock()
	delete(detectModeCache, cameraID)
	detectModeCacheMu.Unlock()
}

type AIWorkerSession struct {
	stream       facerec.FaceInferenceService_ProcessStreamServer
	id           string
	connectedAt  time.Time
	avgProcessMs float64
	isPaused     bool
	mu           sync.Mutex
	sendCh       chan *facerec.FrameTask
	closeOnce    sync.Once
}

// senderLoop drains sendCh and writes to the gRPC stream sequentially.
// gRPC stream.Send is not thread-safe, so all sends must go through here.
func (w *AIWorkerSession) senderLoop() {
	for task := range w.sendCh {
		if err := w.stream.Send(task); err != nil {
			log.Printf("[Worker %s] stream send error: %v — draining queue", w.id, err)
			// drain remaining so dispatcher goroutines don't block on a dead channel
			for range w.sendCh {
			}
			return
		}
	}
}

// closeSendCh signals senderLoop to exit. Safe to call multiple times.
func (w *AIWorkerSession) closeSendCh() {
	w.closeOnce.Do(func() { close(w.sendCh) })
}

// trySend enqueues a task for the worker without blocking the caller.
// Returns false if the worker is overloaded (channel full) or disconnected.
func (w *AIWorkerSession) trySend(task *facerec.FrameTask) (sent bool) {
	defer func() {
		if recover() != nil {
			sent = false // send on closed channel
		}
	}()
	select {
	case w.sendCh <- task:
		return true
	default:
		return false // worker queue full
	}
}

type PendingTask struct {
	CameraID   uint
	Timestamp  int64
	ImageBytes []byte
}

type RegistrationResult struct {
	Embedding    []float32
	ErrorMessage string
}

var (
	// Active gRPC worker streams
	activeWorkers   = make([]*AIWorkerSession, 0)
	activeWorkersMu sync.Mutex
	workerIndex     int

	// Camera-to-Worker mapping for sticky routing
	cameraToWorker   = make(map[uint]string)
	cameraToWorkerMu sync.Mutex

	// Pending tasks waiting for AI response
	pendingTasks   = make(map[string]PendingTask)
	pendingTasksMu sync.Mutex

	// Channels to resolve face registrations
	regChannels   = make(map[string]chan RegistrationResult)
	regChannelsMu sync.Mutex

	// gRPC Server instance
	grpcServer *grpc.Server
)


type FaceInferenceServer struct {
	facerec.UnimplementedFaceInferenceServiceServer
}

func StartGRPCServer() {
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("Failed to listen on port 50051: %v", err)
	}

	grpcServer = grpc.NewServer(
		grpc.MaxRecvMsgSize(20 * 1024 * 1024),
		grpc.MaxSendMsgSize(20 * 1024 * 1024),
	)
	facerec.RegisterFaceInferenceServiceServer(grpcServer, &FaceInferenceServer{})

	log.Println("gRPC Server listening on port 50051...")
	go func() {
		if err := grpcServer.Serve(lis); err != nil {
			log.Printf("gRPC Server failed to serve: %v", err)
		}
	}()

	// Start background frame dispatcher
	go StartFrameDispatcher(context.Background())
}

func (s *FaceInferenceServer) ProcessStream(stream facerec.FaceInferenceService_ProcessStreamServer) error {
	session := &AIWorkerSession{
		stream:      stream,
		id:          uuid.New().String(),
		connectedAt: time.Now(),
		sendCh:      make(chan *facerec.FrameTask, 64),
	}

	go session.senderLoop()
	registerWorker(session)
	defer func() {
		deregisterWorker(session)
		session.closeSendCh()
	}()

	log.Printf("[gRPC] New AI worker connected: %s", session.id)

	for {
		result, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Printf("[gRPC] AI worker disconnected: %s, error: %v", session.id, err)
			break
		}

		// Update average processing time metrics
		if result.ProcessTimeMs > 0 {
			session.mu.Lock()
			session.avgProcessMs = float64(result.ProcessTimeMs)
			session.mu.Unlock()
		}

		// Skip DB & storage tracking for system metric update events
		if result.TaskId == "metrics" {
			continue
		}

		handleInferenceResult(result)
	}

	return nil
}

func registerWorker(w *AIWorkerSession) {
	activeWorkersMu.Lock()
	activeWorkers = append(activeWorkers, w)
	activeWorkersMu.Unlock()

	rebalanceWorkers()
}

func deregisterWorker(w *AIWorkerSession) {
	activeWorkersMu.Lock()
	defer activeWorkersMu.Unlock()
	for i, v := range activeWorkers {
		if v.id == w.id {
			activeWorkers = append(activeWorkers[:i], activeWorkers[i+1:]...)
			break
		}
	}

	// Clean up sticky assignments for this disconnected worker
	cameraToWorkerMu.Lock()
	defer cameraToWorkerMu.Unlock()
	for camID, wID := range cameraToWorker {
		if wID == w.id {
			delete(cameraToWorker, camID)
		}
	}
}

func rebalanceWorkers() {
	activeWorkersMu.Lock()
	numWorkers := 0
	for _, w := range activeWorkers {
		w.mu.Lock()
		if !w.isPaused {
			numWorkers++
		}
		w.mu.Unlock()
	}
	activeWorkersMu.Unlock()

	if numWorkers == 0 {
		cameraToWorkerMu.Lock()
		for k := range cameraToWorker {
			delete(cameraToWorker, k)
		}
		cameraToWorkerMu.Unlock()
		return
	}

	if numWorkers <= 1 {
		return
	}

	cameraToWorkerMu.Lock()
	defer cameraToWorkerMu.Unlock()

	numCameras := len(cameraToWorker)
	if numCameras == 0 {
		return
	}

	// Target limit (ceiling division C / N)
	targetLimit := (numCameras + numWorkers - 1) / numWorkers
	if targetLimit < 1 {
		targetLimit = 1
	}

	log.Printf("[Rebalance] Rebalancing %d cameras across %d active workers (target limit: %d per worker)", numCameras, numWorkers, targetLimit)

	// Group assigned cameras by worker ID
	workerAssignments := make(map[string][]uint)
	for camID, wID := range cameraToWorker {
		workerAssignments[wID] = append(workerAssignments[wID], camID)
	}

	for wID, camIDs := range workerAssignments {
		if len(camIDs) > targetLimit {
			numToRemove := len(camIDs) - targetLimit
			log.Printf("[Rebalance] Worker %s has %d cameras (exceeds limit %d). Unassigning %d camera(s)...", wID, len(camIDs), targetLimit, numToRemove)
			for i := 0; i < numToRemove; i++ {
				camToRemove := camIDs[i]
				delete(cameraToWorker, camToRemove)
				log.Printf("[Rebalance] Unassigned Camera %d from Worker %s", camToRemove, wID)
			}
		}
	}
}

func getWorkerForCamera(cameraID uint) *AIWorkerSession {
	activeWorkersMu.Lock()
	defer activeWorkersMu.Unlock()

	if len(activeWorkers) == 0 {
		return nil
	}

	cameraToWorkerMu.Lock()
	defer cameraToWorkerMu.Unlock()

	// Check if this camera is already assigned to an active, non-paused worker
	if workerID, exists := cameraToWorker[cameraID]; exists {
		// Verify if the assigned worker is still active and not paused
		for _, w := range activeWorkers {
			w.mu.Lock()
			paused := w.isPaused
			w.mu.Unlock()
			if w.id == workerID && !paused {
				return w
			}
		}
		// If worker is no longer active or is paused, remove the stale/paused mapping
		delete(cameraToWorker, cameraID)
	}

	// Calculate current load per active, non-paused worker
	workerLoad := make(map[string]int)
	nonPausedWorkersExist := false
	for _, w := range activeWorkers {
		w.mu.Lock()
		paused := w.isPaused
		w.mu.Unlock()
		if !paused {
			workerLoad[w.id] = 0
			nonPausedWorkersExist = true
		}
	}

	if !nonPausedWorkersExist {
		return nil
	}

	for _, wID := range cameraToWorker {
		if _, ok := workerLoad[wID]; ok {
			workerLoad[wID]++
		}
	}

	// Find the worker with the minimum load
	var bestWorker *AIWorkerSession
	minLoad := int(^uint(0) >> 1) // Max int

	for _, w := range activeWorkers {
		w.mu.Lock()
		paused := w.isPaused
		w.mu.Unlock()
		if paused {
			continue
		}
		load := workerLoad[w.id]
		if load < minLoad {
			minLoad = load
			bestWorker = w
		}
	}

	if bestWorker != nil {
		cameraToWorker[cameraID] = bestWorker.id
		log.Printf("[Dispatcher] Assigned Camera %d to Worker %s (current load: %d cameras)", cameraID, bestWorker.id, minLoad+1)
	}

	return bestWorker
}

func getNextWorker() *AIWorkerSession {
	activeWorkersMu.Lock()
	defer activeWorkersMu.Unlock()
	if len(activeWorkers) == 0 {
		return nil
	}
	// Filter to non-paused workers
	available := make([]*AIWorkerSession, 0)
	for _, w := range activeWorkers {
		w.mu.Lock()
		paused := w.isPaused
		w.mu.Unlock()
		if !paused {
			available = append(available, w)
		}
	}
	if len(available) == 0 {
		return nil
	}
	worker := available[workerIndex%len(available)]
	workerIndex++
	return worker
}

func ToggleWorkerPause(workerID string) (bool, error) {
	activeWorkersMu.Lock()
	var targetWorker *AIWorkerSession
	for _, w := range activeWorkers {
		if w.id == workerID {
			targetWorker = w
			break
		}
	}
	activeWorkersMu.Unlock()

	if targetWorker == nil {
		return false, fmt.Errorf("worker not found")
	}

	targetWorker.mu.Lock()
	targetWorker.isPaused = !targetWorker.isPaused
	newPausedState := targetWorker.isPaused
	targetWorker.mu.Unlock()

	// If paused, clear its camera assignments so they can be re-dispatched
	if newPausedState {
		cameraToWorkerMu.Lock()
		for camID, wID := range cameraToWorker {
			if wID == workerID {
				delete(cameraToWorker, camID)
			}
		}
		cameraToWorkerMu.Unlock()
	}

	// Rebalance active cameras across non-paused workers
	rebalanceWorkers()

	return newPausedState, nil
}

// SendRegistrationTask sends a face image to a worker for embedding extraction
func SendRegistrationTask(ctx context.Context, imgBytes []byte) ([]float32, error) {
	worker := getNextWorker()
	if worker == nil {
		return nil, fmt.Errorf("no AI workers connected")
	}

	taskID := uuid.New().String()
	ch := make(chan RegistrationResult, 1)

	regChannelsMu.Lock()
	regChannels[taskID] = ch
	regChannelsMu.Unlock()
	defer func() {
		regChannelsMu.Lock()
		delete(regChannels, taskID)
		regChannelsMu.Unlock()
	}()

	// Send frame task down the worker stream
	err := worker.stream.Send(&facerec.FrameTask{
		TaskId:         taskID,
		ImageData:      imgBytes,
		IsRegistration: true,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to send task to AI worker: %w", err)
	}

	// Wait for response with a timeout of 10s
	select {
	case res := <-ch:
		if res.ErrorMessage != "" {
			return nil, fmt.Errorf(res.ErrorMessage)
		}
		if len(res.Embedding) == 0 {
			return nil, fmt.Errorf("no face detected in image")
		}
		return res.Embedding, nil
	case <-time.After(10 * time.Second):
		return nil, fmt.Errorf("timeout waiting for embedding extraction")
	}
}

func handleInferenceResult(result *facerec.InferenceResult) {
	// Skip logging metrics completely
	if result.TaskId == "metrics" {
		return
	}

	log.Printf("[gRPC Debug] Received InferenceResult for TaskId=%s with %d detections", result.TaskId, len(result.Detections))

	// 1. Check if it's a registration task
	regChannelsMu.Lock()
	ch, isReg := regChannels[result.TaskId]
	regChannelsMu.Unlock()

	if isReg {
		log.Printf("[gRPC Debug] TaskId=%s is a Registration task", result.TaskId)
		if result.ErrorMessage != "" {
			ch <- RegistrationResult{
				Embedding:    nil,
				ErrorMessage: result.ErrorMessage,
			}
		} else if len(result.Detections) > 0 {
			ch <- RegistrationResult{
				Embedding:    result.Detections[0].Embedding,
				ErrorMessage: "",
			}
		} else {
			ch <- RegistrationResult{
				Embedding:    nil,
				ErrorMessage: "",
			}
		}
		return
	}

	// 2. Otherwise, it's a real-time frame task
	pendingTasksMu.Lock()
	task, exists := pendingTasks[result.TaskId]
	if exists {
		delete(pendingTasks, result.TaskId)
	}
	pendingTasksMu.Unlock()

	if !exists {
		log.Printf("[gRPC Debug] TaskId=%s NOT FOUND in pendingTasks (timed out or invalid)", result.TaskId)
		// Task already timed out or processed
		return
	}

	ctx := context.Background()

	// Handle plate detections (independent of face detections)
	if len(result.PlateDetections) > 0 {
		handlePlateDetections(ctx, result, task)
	}

	// If no faces detected, we skip face logging and S3 uploads
	if len(result.Detections) == 0 {
		log.Printf("[gRPC Debug] TaskId=%s has 0 detections, skipping", result.TaskId)
		return
	}
	
	log.Printf("[gRPC Debug] TaskId=%s matched to Camera %d, proceeding to process", result.TaskId, task.CameraID)

	// Process detections
	type UIResultDetection struct {
		BBox       []float64 `json:"bbox"`
		PersonID   *uint     `json:"person_id"`
		Confidence float64   `json:"confidence"`
	}

	uiDetections := make([]UIResultDetection, 0)
	bboxes := make([][]float64, 0)
	isKnown := make([]bool, 0)

	recorded := 0
	for idx, det := range result.Detections {
		// REST search in Qdrant
		similarityThreshold := 0.4 // settings.SIMILARITY_THRESHOLD
		personID, _, score, err := SearchFaceEmbedding(ctx, det.Embedding, similarityThreshold)
		if err != nil {
			log.Printf("[gRPC] Qdrant search error: %v", err)
		}

		bbox64 := []float64{
			float64(det.Bbox[0]),
			float64(det.Bbox[1]),
			float64(det.Bbox[2]),
			float64(det.Bbox[3]),
		}
		bboxes = append(bboxes, bbox64)
		isKnown = append(isKnown, personID != nil)

		uiDetections = append(uiDetections, UIResultDetection{
			BBox:       bbox64,
			PersonID:   personID,
			Confidence: score,
		})

		logEntry := DetectionLog{
			CameraID:   task.CameraID,
			Confidence: score,
			DetectedAt: time.UnixMilli(task.Timestamp),
			PersonID:   personID,
		}

		// Look up camera name
		var cam Camera
		if DB.First(&cam, task.CameraID).Error == nil {
			logEntry.CameraName = cam.Name
		}

		// Look up person name
		if personID != nil {
			var person Person
			if DB.First(&person, *personID).Error == nil {
				logEntry.PersonName = person.Name
			} else {
				logEntry.PersonName = fmt.Sprintf("Person %d", *personID)
			}
		} else {
			logEntry.PersonName = "Unknown"
		}

		// Save snapshot url path (will draw bboxes first and write it to S3)
		filename := fmt.Sprintf("cam_%d_%d.jpg", task.CameraID, task.Timestamp)
		logEntry.SnapshotPath = "/api/static/snapshots/" + filename

		// Crop the face from the original frame (add 50% padding to match the AI worker crop scale)
		x1, y1, x2, y2 := int(det.Bbox[0]), int(det.Bbox[1]), int(det.Bbox[2]), int(det.Bbox[3])
		padX := (x2 - x1) / 2
		padY := (y2 - y1) / 2
		croppedFace, err := CropJPEG(task.ImageBytes, x1-padX, y1-padY, x2+padX, y2+padY, 90)
		if err == nil && S3Client != nil {
			cropFilename := fmt.Sprintf("crop_cam_%d_%d_%d.jpg", task.CameraID, task.Timestamp, idx)
			_, err = S3Client.PutObject(ctx, SnapshotsBucket, cropFilename, bytes.NewReader(croppedFace), int64(len(croppedFace)), minio.PutObjectOptions{
				ContentType: "image/jpeg",
			})
			if err == nil {
				logEntry.FaceCropPath = "/api/static/snapshots/" + cropFilename
			} else {
				log.Printf("[gRPC] Failed to upload face crop to S3: %v", err)
			}
		} else if err != nil {
			log.Printf("[gRPC] Failed to crop face: %v", err)
		}

		// Upload CodeFormer restored face if present
		if len(det.RestoredFaceJpeg) > 0 && S3Client != nil {
			restoredFilename := fmt.Sprintf("restored_cam_%d_%d_%d.jpg", task.CameraID, task.Timestamp, idx)
			_, err = S3Client.PutObject(ctx, SnapshotsBucket, restoredFilename, bytes.NewReader(det.RestoredFaceJpeg), int64(len(det.RestoredFaceJpeg)), minio.PutObjectOptions{
				ContentType: "image/jpeg",
			})
			if err == nil {
				logEntry.RestoredFacePath = "/api/static/snapshots/" + restoredFilename
				log.Printf("[gRPC] Uploaded restored face crop for camera %d to S3 as %s", task.CameraID, restoredFilename)
			} else {
				log.Printf("[gRPC] Failed to upload restored face to S3: %v", err)
			}
		}

		DB.Create(&logEntry)
		recorded++

		// Broadcast to UI via WebSockets for each detection event
		payload := fiber.Map{
			"type":              "detection",
			"person_id":         personID,
			"person_name":       logEntry.PersonName,
			"camera_id":         task.CameraID,
			"camera_name":       logEntry.CameraName,
			"confidence":        score,
			"snapshot_url":      logEntry.SnapshotPath,
			"face_crop_url":     logEntry.FaceCropPath,
			"restored_face_url": logEntry.RestoredFacePath,
			"timestamp":         time.UnixMilli(task.Timestamp).Format(time.RFC3339),
		}
		BroadcastDetection(payload)
	}

	// 3. Draw bounding boxes on frame copy
	drawnFrame, err := DrawBBoxesOnJPEG(task.ImageBytes, bboxes, isKnown, 70)
	if err != nil {
		log.Printf("[gRPC] Failed to draw bounding boxes on frame: %v", err)
		drawnFrame = task.ImageBytes
	}

	// 4. Upload snapshot to S3
	filename := fmt.Sprintf("cam_%d_%d.jpg", task.CameraID, task.Timestamp)
	if S3Client != nil {
		_, err = S3Client.PutObject(ctx, SnapshotsBucket, filename, bytes.NewReader(drawnFrame), int64(len(drawnFrame)), minio.PutObjectOptions{
			ContentType: "image/jpeg",
		})
		if err != nil {
			log.Printf("[gRPC] Failed to upload snapshot %s to S3: %v", filename, err)
		}
	}

	if recorded > 0 {
		log.Printf("[Camera %d] Recorded %d detections.", task.CameraID, recorded)
	}
}

func StartFrameDispatcher(ctx context.Context) {
	log.Println("Background frame dispatcher started.")

	groupName := "control-plane-group"
	consumerName := "cp-dispatcher"
	streamName := "image.queue"

	// Create consumer group
	RDB.XGroupCreateMkStream(ctx, streamName, groupName, "0")

	for {
		// 1. Wait until we have at least one active AI worker
		for {
			activeWorkersMu.Lock()
			count := len(activeWorkers)
			activeWorkersMu.Unlock()
			if count > 0 {
				break
			}
			time.Sleep(500 * time.Millisecond)
		}

		// 2. Consume a batch of frames from Redis Stream.
		// Reading N at once amortizes the round-trip cost — especially important
		// when multiple cameras push frames concurrently.
		res, err := RDB.XReadGroup(ctx, &redis.XReadGroupArgs{
			Group:    groupName,
			Consumer: consumerName,
			Streams:  []string{streamName, ">"},
			Count:    10,
			Block:    2000 * time.Millisecond,
		}).Result()

		if err != nil {
			if err != redis.Nil {
				log.Printf("[Dispatcher] Redis Stream read error: %v", err)
			}
			continue
		}

		for _, stream := range res {
			for _, msg := range stream.Messages {
				var cameraID uint
				var ts int64
				var data []byte

				// Safe parsing camera_id
				if val, ok := msg.Values["camera_id"].(string); ok {
					cid, _ := fmt.Sscanf(val, "%d", &cameraID)
					if cid == 0 {
						// fallback
						var valInt int
						fmt.Sscanf(val, "%d", &valInt)
						cameraID = uint(valInt)
					}
				} else if val, ok := msg.Values["camera_id"].(int64); ok {
					cameraID = uint(val)
				}

				// Safe parsing ts
				if val, ok := msg.Values["ts"].(string); ok {
					fmt.Sscanf(val, "%d", &ts)
				} else if val, ok := msg.Values["ts"].(int64); ok {
					ts = val
				}

				// Safe parsing data
				if val, ok := msg.Values["data"].(string); ok {
					data = []byte(val)
				} else if val, ok := msg.Values["data"].([]byte); ok {
					data = val
				}

				// Dispatch sticky worker by camera ID
				worker := getWorkerForCamera(cameraID)
				if worker == nil {
					log.Printf("[Dispatcher] No worker available for camera %d, skipped frame.", cameraID)
					continue
				}

				taskID := fmt.Sprintf("%d_%d_%s", cameraID, ts, uuid.New().String())

				// Register pending task
				pendingTasksMu.Lock()
				pendingTasks[taskID] = PendingTask{
					CameraID:   cameraID,
					Timestamp:  ts,
					ImageBytes: data,
				}
				pendingTasksMu.Unlock()

				// Clean up pending task after timeout using a timer (avoids spawning
				// a goroutine per frame, which would accumulate thousands of sleeping goroutines).
				time.AfterFunc(30*time.Second, func() {
					pendingTasksMu.Lock()
					delete(pendingTasks, taskID)
					pendingTasksMu.Unlock()
				})

				// Look up camera detect_mode from in-memory cache (refreshes every 8 s).
				detectMode := getCachedDetectMode(cameraID)

				// Acknowledge Redis Stream message before dispatch so the
				// read loop is never blocked by a slow or overloaded worker.
				RDB.XAck(ctx, streamName, groupName, msg.ID)
				RDB.XDel(ctx, streamName, msg.ID)

				// Enqueue frame to the worker's send channel (non-blocking).
				// senderLoop owns the stream.Send call, so this is thread-safe.
				sent := worker.trySend(&facerec.FrameTask{
					TaskId:         taskID,
					ImageData:      data,
					IsRegistration: false,
					DetectMode:     detectMode,
				})
				if !sent {
					log.Printf("[Dispatcher] Worker %s overloaded or disconnected — dropping frame for camera %d", worker.id, cameraID)
					pendingTasksMu.Lock()
					delete(pendingTasks, taskID)
					pendingTasksMu.Unlock()
				}
			}
		}
	}
}
