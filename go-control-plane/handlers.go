package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/url"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/websocket/v2"
	"github.com/google/uuid"
	"github.com/minio/minio-go/v7"
)

// --- Camera Handlers ---

func GetCameras(c *fiber.Ctx) error {
	cameras := make([]Camera, 0)
	DB.Find(&cameras)
	return c.JSON(fiber.Map{
		"items": cameras,
		"total": len(cameras),
	})
}

func AddCamera(c *fiber.Ctx) error {
	camera := new(Camera)
	if err := c.BodyParser(camera); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": err.Error()})
	}
	camera.CreatedAt = time.Now()
	DB.Create(&camera)
	return c.Status(201).JSON(camera)
}

func UpdateCamera(c *fiber.Ctx) error {
	id := c.Params("id")
	var camera Camera
	if err := DB.First(&camera, id).Error; err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "Camera not found"})
	}

	if err := c.BodyParser(&camera); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": err.Error()})
	}

	DB.Save(&camera)
	InvalidateDetectModeCache(camera.ID)
	return c.JSON(camera)
}

func DeleteCamera(c *fiber.Ctx) error {
	id := c.Params("id")
	DB.Delete(&Camera{}, id)
	return c.SendStatus(204)
}

func StartCameraStream(c *fiber.Ctx) error {
	idStr := c.Params("id")
	id, err := strconv.ParseUint(idStr, 10, 32)
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "Invalid ID"})
	}

	var camera Camera
	if err := DB.First(&camera, id).Error; err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "Camera not found"})
	}

	fps := camera.FPSProcess
	if fps <= 0 {
		fps = 2 // default fallback to prevent division by zero in the ingestion worker
	}

	if err := PublishAssignment(uint(id), camera.URL, fps); err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "Failed to publish start command"})
	}

	DB.Model(&camera).Update("IsActive", true)
	return c.JSON(fiber.Map{"status": "started", "camera_id": id})
}

func StopCameraStream(c *fiber.Ctx) error {
	idStr := c.Params("id")
	id, err := strconv.ParseUint(idStr, 10, 32)
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "Invalid ID"})
	}

	var camera Camera
	if err := DB.First(&camera, id).Error; err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "Camera not found"})
	}

	if err := PublishStopCommand(uint(id)); err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "Failed to publish stop command"})
	}

	DB.Model(&camera).Update("IsActive", false)
	return c.JSON(fiber.Map{"status": "stopped", "camera_id": id})
}

func GetCameraSnapshot(c *fiber.Ctx) error {
	idStr := c.Params("id")
	frame, err := RDB.Get(ctx, "camera:latest:"+idStr).Bytes()
	if err != nil {
		return c.Status(404).SendString("Live snapshot not available")
	}
	c.Set("Content-Type", "image/jpeg")
	return c.Send(frame)
}

// --- Person Handlers ---

func GetPersons(c *fiber.Ctx) error {
	search := c.Query("search", "")
	page, _ := strconv.Atoi(c.Query("page", "1"))
	limit, _ := strconv.Atoi(c.Query("limit", "20"))
	offset := (page - 1) * limit

	persons := make([]Person, 0)
	var total int64
	query := DB.Model(&Person{})
	if search != "" {
		query = query.Where("name ILIKE ?", "%"+search+"%")
	}
	query.Count(&total)
	query.Preload("Faces").Order("created_at desc").Offset(offset).Limit(limit).Find(&persons)

	return c.JSON(fiber.Map{
		"items": persons,
		"total": total,
		"page":  page,
		"limit": limit,
	})
}

func CreatePerson(c *fiber.Ctx) error {
	person := new(Person)
	if err := c.BodyParser(person); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": err.Error()})
	}
	person.CreatedAt = time.Now()
	person.UpdatedAt = time.Now()
	DB.Create(&person)
	return c.Status(201).JSON(person)
}

func GetPerson(c *fiber.Ctx) error {
	id := c.Params("id")
	var person Person
	if err := DB.Preload("Faces").First(&person, id).Error; err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "Person not found"})
	}
	return c.JSON(person)
}

func UpdatePerson(c *fiber.Ctx) error {
	id := c.Params("id")
	var person Person
	if err := DB.First(&person, id).Error; err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "Person not found"})
	}
	if err := c.BodyParser(&person); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": err.Error()})
	}
	person.UpdatedAt = time.Now()
	DB.Save(&person)
	return c.JSON(person)
}

func DeletePerson(c *fiber.Ctx) error {
	idStr := c.Params("id")
	personID, err := strconv.ParseUint(idStr, 10, 32)
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "Invalid person ID"})
	}

	// 1. Get all faces of this person to delete from S3
	var faces []PersonFace
	DB.Where("person_id = ?", personID).Find(&faces)

	// 2. Delete face images from S3
	if S3Client != nil {
		for _, face := range faces {
			err := S3Client.RemoveObject(c.Context(), FacesBucket, face.ImagePath, minio.RemoveObjectOptions{})
			if err != nil {
				log.Printf("[Person %d] Failed to delete S3 face image %s: %v", personID, face.ImagePath, err)
			}
		}
	}

	// 3. Delete from Qdrant — must succeed before removing DB records, otherwise
	// embeddings become orphaned and the person keeps triggering detections.
	err = DeletePersonEmbeddings(c.Context(), uint(personID))
	if err != nil {
		log.Printf("[Person %d] Failed to delete embeddings from Qdrant: %v", personID, err)
		return c.Status(500).JSON(fiber.Map{"error": "Failed to delete face embeddings from vector database"})
	}

	// 4. Delete person and cascade faces in database
	DB.Where("person_id = ?", personID).Delete(&PersonFace{})
	DB.Delete(&Person{}, personID)

	return c.SendStatus(204)
}

func UploadFace(c *fiber.Ctx) error {
	personID, _ := strconv.ParseUint(c.Params("id"), 10, 32)

	form, err := c.MultipartForm()
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "Invalid form data"})
	}

	files := form.File["files"]
	if len(files) == 0 {
		// Also try singular "file" for backward compatibility
		files = form.File["file"]
	}
	if len(files) == 0 {
		return c.Status(400).JSON(fiber.Map{"error": "No files uploaded"})
	}

	var createdFaces []PersonFace

	for _, file := range files {
		ext := filepath.Ext(file.Filename)
		filename := fmt.Sprintf("%d_%s%s", personID, uuid.New().String()[:8], ext)

		// Open the file
		fileHeader, err := file.Open()
		if err != nil {
			log.Printf("[Person %d] Failed to open uploaded file %s: %v", personID, file.Filename, err)
			continue
		}
		imgBytes, err := io.ReadAll(fileHeader)
		fileHeader.Close()
		if err != nil {
			log.Printf("[Person %d] Failed to read uploaded file %s: %v", personID, file.Filename, err)
			continue
		}

		// 1. Ask AI worker via gRPC to extract embedding
		embedding, err := SendRegistrationTask(c.Context(), imgBytes)
		if err != nil {
			log.Printf("[Person %d] Face extraction failed for %s: %v", personID, file.Filename, err)
			return c.Status(400).JSON(fiber.Map{"error": fmt.Sprintf("Face extraction failed: %v", err)})
		}

		// 2. Upload original file to S3
		if S3Client != nil {
			_, err = S3Client.PutObject(c.Context(), FacesBucket, filename, bytes.NewReader(imgBytes), int64(len(imgBytes)), minio.PutObjectOptions{
				ContentType: file.Header.Get("Content-Type"),
			})
			if err != nil {
				log.Printf("[Person %d] Failed to upload face image to S3: %v", personID, err)
				return c.Status(500).JSON(fiber.Map{"error": fmt.Sprintf("Failed to store face image: %v", err)})
			}
		}

		// 3. Save face metadata & embedding in DB
		face := PersonFace{
			PersonID:  uint(personID),
			ImagePath: filename,
			Embedding: float32SliceToBytes(embedding),
			CreatedAt: time.Now(),
		}
		DB.Create(&face)

		// 4. Save embedding to Qdrant directly
		err = AddFaceEmbedding(c.Context(), uint(personID), face.ID, embedding)
		if err != nil {
			log.Printf("[Person %d] Failed to save face ID %d to Qdrant: %v", personID, face.ID, err)
		}

		createdFaces = append(createdFaces, face)
	}

	return c.Status(201).JSON(createdFaces)
}

func DeleteFace(c *fiber.Ctx) error {
	personID, _ := strconv.ParseUint(c.Params("id"), 10, 32)
	faceID, _ := strconv.ParseUint(c.Params("face_id"), 10, 32)

	var face PersonFace
	if err := DB.Where("id = ? AND person_id = ?", faceID, personID).First(&face).Error; err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "Face not found"})
	}

	// 1. Delete image from S3
	if S3Client != nil {
		err := S3Client.RemoveObject(c.Context(), FacesBucket, face.ImagePath, minio.RemoveObjectOptions{})
		if err != nil {
			log.Printf("[Person %d] Failed to delete S3 face image %s: %v", personID, face.ImagePath, err)
		}
	}

	// 2. Delete from Qdrant directly
	err := DeleteFaceEmbedding(c.Context(), uint(faceID))
	if err != nil {
		log.Printf("[Person %d] Failed to delete face ID %d from Qdrant: %v", personID, faceID, err)
	}

	// 3. Delete from DB
	DB.Delete(&face)

	return c.SendStatus(204)
}

func float32SliceToBytes(slice []float32) []byte {
	if len(slice) == 0 {
		return nil
	}
	buf := new(bytes.Buffer)
	binary.Write(buf, binary.LittleEndian, slice)
	return buf.Bytes()
}


// --- Detection Handlers ---

func listDetections(c *fiber.Ctx) error {
	page, _ := strconv.Atoi(c.Query("page", "1"))
	limit, _ := strconv.Atoi(c.Query("limit", "20"))
	offset := (page - 1) * limit

	query := DB.Model(&DetectionLog{})

	if personIDStr := c.Query("person_id"); personIDStr != "" {
		if personID, err := strconv.Atoi(personIDStr); err == nil {
			query = query.Where("person_id = ?", personID)
		} else if personIDStr == "null" {
			query = query.Where("person_id IS NULL")
		}
	}

	if cameraIDStr := c.Query("camera_id"); cameraIDStr != "" {
		if cameraID, err := strconv.Atoi(cameraIDStr); err == nil {
			query = query.Where("camera_id = ?", cameraID)
		}
	}

	if dateFromStr := c.Query("date_from"); dateFromStr != "" {
		if dateFrom, err := time.Parse(time.RFC3339, dateFromStr); err == nil {
			query = query.Where("detected_at >= ?", dateFrom)
		} else if dateFrom, err := time.Parse("2006-01-02", dateFromStr); err == nil {
			query = query.Where("detected_at >= ?", dateFrom)
		}
	}

	if dateToStr := c.Query("date_to"); dateToStr != "" {
		if dateTo, err := time.Parse(time.RFC3339, dateToStr); err == nil {
			query = query.Where("detected_at <= ?", dateTo)
		} else if dateTo, err := time.Parse("2006-01-02", dateToStr); err == nil {
			query = query.Where("detected_at <= ?", dateTo.Add(24*time.Hour))
		}
	}

	detections := make([]DetectionLog, 0)
	var total int64
	query.Count(&total)
	query.Order("detected_at desc").Offset(offset).Limit(limit).Find(&detections)

	return c.JSON(fiber.Map{
		"items": detections,
		"total": total,
		"page":  page,
		"limit": limit,
	})
}

func getDetectionStats(c *fiber.Ctx) error {
	queryLog := DB.Model(&DetectionLog{})

	if dateFromStr := c.Query("date_from"); dateFromStr != "" {
		if dateFrom, err := time.Parse(time.RFC3339, dateFromStr); err == nil {
			queryLog = queryLog.Where("detected_at >= ?", dateFrom)
		} else if dateFrom, err := time.Parse("2006-01-02", dateFromStr); err == nil {
			queryLog = queryLog.Where("detected_at >= ?", dateFrom)
		}
	}

	if dateToStr := c.Query("date_to"); dateToStr != "" {
		if dateTo, err := time.Parse(time.RFC3339, dateToStr); err == nil {
			queryLog = queryLog.Where("detected_at <= ?", dateTo)
		} else if dateTo, err := time.Parse("2006-01-02", dateToStr); err == nil {
			queryLog = queryLog.Where("detected_at <= ?", dateTo.Add(24*time.Hour))
		}
	}

	var total int64
	var uniquePersons int64
	queryLog.Count(&total)
	queryLog.Distinct("person_id").Count(&uniquePersons)

	type CameraStat struct {
		CameraName string `json:"camera_name"`
		Count      int64  `json:"count"`
	}
	var cameraStats []CameraStat
	queryLog.Select("camera_name, count(id) as count").Group("camera_name").Scan(&cameraStats)

	byCamera := make(map[string]int64)
	for _, s := range cameraStats {
		byCamera[s.CameraName] = s.Count
	}

	return c.JSON(fiber.Map{
		"total_detections": total,
		"unique_persons":   uniquePersons,
		"by_camera":        byCamera,
		"by_hour":          make(map[string]int),
	})
}

func getOverviewStats(c *fiber.Ctx) error {
	var totalCameras int64
	var activeCameras int64
	var totalPersons int64
	var detectionsToday int64
	DB.Model(&Camera{}).Count(&totalCameras)
	DB.Model(&Camera{}).Where("is_active = ?", true).Count(&activeCameras)
	DB.Model(&Person{}).Count(&totalPersons)
	today := time.Now().Truncate(24 * time.Hour)
	DB.Model(&DetectionLog{}).Where("detected_at >= ?", today).Count(&detectionsToday)

	return c.JSON(fiber.Map{
		"total_cameras":          totalCameras,
		"active_cameras":         activeCameras,
		"total_persons":          totalPersons,
		"total_detections_today": detectionsToday,
	})
}

// --- WebSocket logic ---

var (
	clients   = make(map[*websocket.Conn]bool)
	clientsMu sync.Mutex
)

func EventsWebSocket(c *websocket.Conn) {
	clientsMu.Lock()
	clients[c] = true
	clientsMu.Unlock()
	defer func() {
		clientsMu.Lock()
		delete(clients, c)
		clientsMu.Unlock()
		c.Close()
	}()
	for {
		if _, _, err := c.ReadMessage(); err != nil {
			break
		}
	}
}

func BroadcastDetection(payload interface{}) {
	msg, _ := json.Marshal(payload)
	clientsMu.Lock()
	defer clientsMu.Unlock()
	for client := range clients {
		if err := client.WriteMessage(websocket.TextMessage, msg); err != nil {
			log.Printf("WebSocket broadcast error: %v", err)
			client.Close()
			delete(clients, client)
		}
	}
}

// --- Surveillance Station Stubs ---

// --- Surveillance Station Handlers ---

type SSConnectRequest struct {
	BaseURL   string `json:"base_url"`
	Username  string `json:"username"`
	Password  string `json:"password"`
	VerifySSL bool   `json:"verify_ssl"`
}

type SSImportRequest struct {
	SSConnectRequest
	CameraIDs []int `json:"camera_ids"`
}

func TestSSConnection(c *fiber.Ctx) error {
	var req SSConnectRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "Invalid request body"})
	}

	client := NewSSClient(req.BaseURL, req.Username, req.Password, req.VerifySSL)
	if err := client.Login(); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": fmt.Sprintf("Authentication failed: %v", err)})
	}
	defer client.Logout()

	cameras, err := client.ListCameras()
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": fmt.Sprintf("Failed to list cameras: %v", err)})
	}

	return c.JSON(fiber.Map{
		"status":       "connected",
		"message":      fmt.Sprintf("Successfully connected. Found %d cameras.", len(cameras)),
		"camera_count": len(cameras),
	})
}

func ListSSCameras(c *fiber.Ctx) error {
	var req SSConnectRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "Invalid request body"})
	}

	client := NewSSClient(req.BaseURL, req.Username, req.Password, req.VerifySSL)
	if err := client.Login(); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": fmt.Sprintf("Authentication failed: %v", err)})
	}
	defer client.Logout()

	cameras, err := client.ListCameras()
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": fmt.Sprintf("Failed to list cameras: %v", err)})
	}

	// Figure out NAS host for RTSP Construction
	parsedURL, err := url.Parse(req.BaseURL)
	nasHost := req.BaseURL
	if err == nil && parsedURL.Hostname() != "" {
		nasHost = parsedURL.Hostname()
	} else {
		nasHost = strings.Split(strings.TrimPrefix(strings.TrimPrefix(req.BaseURL, "https://"), "http://"), ":")[0]
	}

	// Check existing
	var existingURLs []string
	DB.Model(&Camera{}).Pluck("url", &existingURLs)
	existingMap := make(map[string]bool)
	for _, u := range existingURLs {
		existingMap[u] = true
	}

	responseCameras := make([]fiber.Map, 0)
	for _, cam := range cameras {
		rtspURL := fmt.Sprintf("ffmpeg:rtsp://dummy:dummy@%s:554/camId=%d&_sid=%s#video=copy#audio=copy#rtsp_transport=tcp", nasHost, cam.ID, client.SID)

		responseCameras = append(responseCameras, fiber.Map{
			"ss_id":            cam.ID,
			"name":             cam.Name,
			"model":            cam.Model,
			"host":             cam.Host,
			"port":             cam.Port,
			"status":           cam.Status,
			"enabled":          cam.Enabled,
			"vendor":           cam.Vendor,
			"resolution":       cam.Resolution,
			"rtsp_url":         rtspURL,
			"already_imported": existingMap[rtspURL],
		})
	}

	return c.JSON(fiber.Map{
		"cameras": responseCameras,
		"total":   len(responseCameras),
		"nas_url": req.BaseURL,
	})
}

func ImportSSCameras(c *fiber.Ctx) error {
	var req SSImportRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "Invalid request body"})
	}

	client := NewSSClient(req.BaseURL, req.Username, req.Password, req.VerifySSL)
	if err := client.Login(); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": fmt.Sprintf("Authentication failed: %v", err)})
	}
	// DO NOT defer logout here! The SID must remain valid if embedded in the imported RTSP IPs momentarily.
	// But actually, Synology checks active SID. A background worker periodically pulling from the RTSP might fail later.
	// However, we just port identical behavior to Python.

	cameras, err := client.ListCameras()
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": fmt.Sprintf("Failed to list cameras: %v", err)})
	}

	camMap := make(map[int]SSCamera)
	for _, cam := range cameras {
		camMap[cam.ID] = cam
	}

	parsedURL, _ := url.Parse(req.BaseURL)
	nasHost := req.BaseURL
	if parsedURL != nil && parsedURL.Hostname() != "" {
		nasHost = parsedURL.Hostname()
	} else {
		nasHost = strings.Split(strings.TrimPrefix(strings.TrimPrefix(req.BaseURL, "https://"), "http://"), ":")[0]
	}

	var existingURLs []string
	DB.Model(&Camera{}).Pluck("url", &existingURLs)
	existingMap := make(map[string]bool)
	for _, u := range existingURLs {
		existingMap[u] = true
	}

	imported := 0
	skipped := 0
	errors := make([]string, 0)
	createdCameras := make([]Camera, 0)

	for _, ssID := range req.CameraIDs {
		cam, found := camMap[ssID]
		if !found {
			errors = append(errors, fmt.Sprintf("Camera ID %d not found", ssID))
			continue
		}

		rtspURL := fmt.Sprintf("ffmpeg:rtsp://dummy:dummy@%s:554/camId=%d&_sid=%s#video=h264#audio=copy#rtsp_transport=tcp", nasHost, cam.ID, client.SID)

		if existingMap[rtspURL] {
			skipped++
			continue
		}

		location := fmt.Sprintf("%s %s (%s)", cam.Vendor, cam.Model, cam.Host)
		if strings.TrimSpace(location) == "()" {
			location = ""
		}

		newCam := Camera{
			Name:       cam.Name,
			URL:        rtspURL,
			Location:   location,
			FPSProcess: 2,
			CreatedAt:  time.Now(),
		}

		if err := DB.Create(&newCam).Error; err != nil {
			errors = append(errors, fmt.Sprintf("Failed to insert %s in DB", cam.Name))
			continue
		}

		createdCameras = append(createdCameras, newCam)
		existingMap[rtspURL] = true
		imported++
	}

	return c.JSON(fiber.Map{
		"imported": imported,
		"skipped":  skipped,
		"errors":   errors,
		"cameras":  createdCameras,
	})
}

// GetWorkers returns the list of active AI Workers and their assigned cameras.
func GetWorkers(c *fiber.Ctx) error {
	activeWorkersMu.Lock()
	workersCopy := make([]*AIWorkerSession, len(activeWorkers))
	copy(workersCopy, activeWorkers)
	activeWorkersMu.Unlock()

	cameraToWorkerMu.Lock()
	camMapCopy := make(map[uint]string)
	for k, v := range cameraToWorker {
		camMapCopy[k] = v
	}
	cameraToWorkerMu.Unlock()

	// Load cameras to map IDs to Names
	var cameras []Camera
	DB.Find(&cameras)
	cameraNames := make(map[uint]string)
	for _, cam := range cameras {
		cameraNames[cam.ID] = cam.Name
	}

	type WorkerCameraInfo struct {
		ID   uint   `json:"id"`
		Name string `json:"name"`
	}

	type WorkerInfo struct {
		ID           string             `json:"id"`
		Name         string             `json:"name"`
		ConnectedAt  string             `json:"connected_at"`
		Uptime       string             `json:"uptime"`
		Cameras      []WorkerCameraInfo `json:"cameras"`
		AvgProcessMs float64            `json:"avg_process_ms"`
		IsPaused     bool               `json:"is_paused"`
	}

	result := make([]WorkerInfo, 0)
	for _, w := range workersCopy {
		assignedCams := make([]WorkerCameraInfo, 0)
		for camID, workerID := range camMapCopy {
			if workerID == w.id {
				name, exists := cameraNames[camID]
				if !exists {
					name = fmt.Sprintf("Camera %d", camID)
				}
				assignedCams = append(assignedCams, WorkerCameraInfo{
					ID:   camID,
					Name: name,
				})
			}
		}

		uptime := time.Since(w.connectedAt).Truncate(time.Second).String()

		w.mu.Lock()
		avgProcessMs := w.avgProcessMs
		isPaused := w.isPaused
		w.mu.Unlock()

		workerName := getSettingValue("worker_name_" + w.id)

		result = append(result, WorkerInfo{
			ID:           w.id,
			Name:         workerName,
			ConnectedAt:  w.connectedAt.Format(time.RFC3339),
			Uptime:       uptime,
			Cameras:      assignedCams,
			AvgProcessMs: avgProcessMs,
			IsPaused:     isPaused,
		})
	}

	return c.JSON(fiber.Map{
		"workers": result,
		"total":   len(result),
	})
}

func RenameWorkerHandler(c *fiber.Ctx) error {
	id := c.Params("id")
	var body struct {
		Name string `json:"name"`
	}
	if err := c.BodyParser(&body); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid body"})
	}
	putSettingValue("worker_name_"+id, body.Name)
	return c.JSON(fiber.Map{"id": id, "name": body.Name})
}

// ToggleWorkerPauseHandler toggles the pause/resume state of an AI worker session.
func ToggleWorkerPauseHandler(c *fiber.Ctx) error {
	id := c.Params("id")
	isPaused, err := ToggleWorkerPause(id)
	if err != nil {
		return c.Status(404).JSON(fiber.Map{"error": err.Error()})
	}
	return c.JSON(fiber.Map{"id": id, "is_paused": isPaused})
}

