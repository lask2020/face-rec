package main

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"strconv"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/minio/minio-go/v7"

	facerec "github.com/face-rec/go-control-plane/facerec"
)

// ── REST Handlers ─────────────────────────────────────────────────────────────

func listPlateDetections(c *fiber.Ctx) error {
	page, _ := strconv.Atoi(c.Query("page", "1"))
	limit, _ := strconv.Atoi(c.Query("limit", "20"))
	cameraID := c.Query("camera_id", "")
	dateFrom := c.Query("date_from", "")
	dateTo := c.Query("date_to", "")

	if page < 1 {
		page = 1
	}
	if limit < 1 || limit > 100 {
		limit = 20
	}
	offset := (page - 1) * limit

	query := DB.Model(&PlateDetectionLog{})

	if cameraID != "" {
		query = query.Where("camera_id = ?", cameraID)
	}
	if dateFrom != "" {
		query = query.Where("detected_at >= ?", dateFrom)
	}
	if dateTo != "" {
		query = query.Where("detected_at <= ?", dateTo)
	}

	var total int64
	query.Count(&total)

	var logs []PlateDetectionLog
	query.Order("detected_at DESC").Limit(limit).Offset(offset).Find(&logs)

	// Resolve snapshot URLs
	for i := range logs {
		if logs[i].SnapshotPath != "" {
			logs[i].SnapshotPath = logs[i].SnapshotPath
		}
	}

	return c.JSON(fiber.Map{
		"items": logs,
		"total": total,
		"page":  page,
		"limit": limit,
	})
}

func getPlateDetectionStats(c *fiber.Ctx) error {
	type DailyStat struct {
		Date  string `json:"date"`
		Count int64  `json:"count"`
	}

	var totalToday int64
	today := time.Now().Truncate(24 * time.Hour)
	DB.Model(&PlateDetectionLog{}).
		Where("detected_at >= ?", today).
		Count(&totalToday)

	var totalAll int64
	DB.Model(&PlateDetectionLog{}).Count(&totalAll)

	type CameraStat struct {
		CameraName string `json:"camera_name"`
		Count      int64  `json:"count"`
	}
	var byCamera []CameraStat
	DB.Model(&PlateDetectionLog{}).
		Select("camera_name, count(*) as count").
		Group("camera_name").
		Find(&byCamera)

	return c.JSON(fiber.Map{
		"total_today": totalToday,
		"total_all":   totalAll,
		"by_camera":   byCamera,
	})
}

// ── gRPC result handler ───────────────────────────────────────────────────────

func handlePlateDetections(ctx context.Context, result *facerec.InferenceResult, task PendingTask) {
	var cam Camera
	DB.First(&cam, task.CameraID)

	for idx, pd := range result.PlateDetections {
		logEntry := PlateDetectionLog{
			CameraID:    task.CameraID,
			CameraName:  cam.Name,
			PlateNumber: pd.PlateNumber,
			RawText:     pd.RawText,
			Confidence:  float64(pd.Confidence),
			PlateType:   pd.PlateType,
			Province:    pd.Province,
			DetectedAt:  time.UnixMilli(task.Timestamp),
		}

		if S3Client != nil {
			filename := fmt.Sprintf("plate_cam_%d_%d_%d.jpg", task.CameraID, task.Timestamp, idx)

			// Crop around the plate bbox with generous padding so the car is visible.
			// Padding: 3× the plate height vertically, 1.5× the plate width horizontally.
			imgBytes := task.ImageBytes
			if len(pd.Bbox) == 4 {
				x1 := int(pd.Bbox[0])
				y1 := int(pd.Bbox[1])
				x2 := int(pd.Bbox[2])
				y2 := int(pd.Bbox[3])
				pw := int(float64(x2-x1) * 1.5)
				ph := int(float64(y2-y1) * 3.0)
				cropped, err := CropJPEG(task.ImageBytes, x1-pw, y1-ph, x2+pw, y2+ph, 88)
				if err == nil {
					imgBytes = cropped
				}
			}

			_, err := S3Client.PutObject(
				ctx, SnapshotsBucket, filename,
				bytes.NewReader(imgBytes), int64(len(imgBytes)),
				minio.PutObjectOptions{ContentType: "image/jpeg"},
			)
			if err == nil {
				logEntry.SnapshotPath = "/api/static/snapshots/" + filename
			} else {
				log.Printf("[Plate] Failed to upload snapshot to S3: %v", err)
			}
		}

		DB.Create(&logEntry)

		BroadcastDetection(fiber.Map{
			"type":         "plate_detection",
			"camera_id":    task.CameraID,
			"camera_name":  cam.Name,
			"plate_number": pd.PlateNumber,
			"raw_text":     pd.RawText,
			"confidence":   pd.Confidence,
			"plate_type":   pd.PlateType,
			"province":     pd.Province,
			"snapshot_url": logEntry.SnapshotPath,
			"timestamp":    time.UnixMilli(task.Timestamp).Format(time.RFC3339),
		})
	}

	log.Printf("[Camera %d] Recorded %d plate detections.", task.CameraID, len(result.PlateDetections))
}
