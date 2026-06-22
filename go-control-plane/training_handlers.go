package main

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"strconv"
	"strings"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/minio/minio-go/v7"

	facerec "github.com/face-rec/go-control-plane/facerec"
)

// ── S3 Save ───────────────────────────────────────────────────────────────────

func saveTrainingFrames(ctx context.Context, frames []*facerec.PlateTrainingFrame, task PendingTask) {
	var cam Camera
	DB.First(&cam, task.CameraID)

	for idx, frame := range frames {
		if len(frame.CropJpeg) == 0 {
			continue
		}

		sample := PlateTrainingSample{
			CameraID:   task.CameraID,
			CameraName: cam.Name,
			CharLabels: frame.CharLabelsJson,
			RawText:    frame.RawText,
			Confidence: float64(frame.Confidence),
			Status:     "pending",
			DetectedAt: time.UnixMilli(task.Timestamp),
		}

		if S3Client != nil {
			filename := fmt.Sprintf("training_cam%d_%d_%d.jpg", task.CameraID, task.Timestamp, idx)
			_, err := S3Client.PutObject(
				ctx, SnapshotsBucket, filename,
				bytes.NewReader(frame.CropJpeg), int64(len(frame.CropJpeg)),
				minio.PutObjectOptions{ContentType: "image/jpeg"},
			)
			if err == nil {
				sample.ImagePath = filename
			} else {
				log.Printf("[Training] S3 upload failed: %v", err)
			}
		}

		if err := DB.Create(&sample).Error; err != nil {
			log.Printf("[Training] DB save failed: %v", err)
		}
	}
}

// ── REST Handlers ─────────────────────────────────────────────────────────────

func listTrainingSamples(c *fiber.Ctx) error {
	page, _ := strconv.Atoi(c.Query("page", "1"))
	limit, _ := strconv.Atoi(c.Query("limit", "20"))
	status := c.Query("status", "")
	confMaxStr := c.Query("conf_max", "")
	confMinStr := c.Query("conf_min", "")
	cameraID := c.Query("camera_id", "")

	if page < 1 {
		page = 1
	}
	if limit < 1 || limit > 100 {
		limit = 20
	}
	offset := (page - 1) * limit

	query := DB.Model(&PlateTrainingSample{})
	if status != "" {
		query = query.Where("status = ?", status)
	}
	if confMaxStr != "" {
		if v, err := strconv.ParseFloat(confMaxStr, 64); err == nil {
			query = query.Where("confidence <= ?", v)
		}
	}
	if confMinStr != "" {
		if v, err := strconv.ParseFloat(confMinStr, 64); err == nil {
			query = query.Where("confidence >= ?", v)
		}
	}
	if cameraID != "" {
		query = query.Where("camera_id = ?", cameraID)
	}

	var total int64
	query.Count(&total)

	var samples []PlateTrainingSample
	// Active-learning sort: lowest confidence first so reviewers see hardest cases first
	query.Order("confidence ASC, detected_at DESC").Limit(limit).Offset(offset).Find(&samples)

	return c.JSON(fiber.Map{
		"items": samples,
		"total": total,
		"page":  page,
		"limit": limit,
	})
}

func getTrainingSample(c *fiber.Ctx) error {
	id := c.Params("id")
	var sample PlateTrainingSample
	if err := DB.First(&sample, id).Error; err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "not found"})
	}
	return c.JSON(sample)
}

func updateTrainingSample(c *fiber.Ctx) error {
	id := c.Params("id")
	var sample PlateTrainingSample
	if err := DB.First(&sample, id).Error; err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "not found"})
	}

	var body struct {
		Status        string `json:"status"`
		CorrectedText string `json:"corrected_text"`
		CharLabels    string `json:"char_labels"`
	}
	if err := c.BodyParser(&body); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": err.Error()})
	}

	updates := map[string]interface{}{}
	if body.Status != "" {
		updates["status"] = body.Status
	}
	if body.CorrectedText != "" {
		updates["corrected_text"] = body.CorrectedText
	}
	if body.CharLabels != "" {
		updates["char_labels"] = body.CharLabels
	}
	DB.Model(&sample).Updates(updates)

	return c.JSON(sample)
}

func bulkUpdateTrainingSamples(c *fiber.Ctx) error {
	var body struct {
		IDs           []uint `json:"ids"`
		Status        string `json:"status"`
		CorrectedText string `json:"corrected_text"`
	}
	if err := c.BodyParser(&body); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": err.Error()})
	}
	if len(body.IDs) == 0 {
		return c.Status(400).JSON(fiber.Map{"error": "ids required"})
	}

	updates := map[string]interface{}{}
	if body.Status != "" {
		updates["status"] = body.Status
	}
	if body.CorrectedText != "" {
		updates["corrected_text"] = body.CorrectedText
	}
	result := DB.Model(&PlateTrainingSample{}).Where("id IN ?", body.IDs).Updates(updates)
	return c.JSON(fiber.Map{"updated": result.RowsAffected})
}

func getTrainingStats(c *fiber.Ctx) error {
	type StatusCount struct {
		Status string `json:"status"`
		Count  int64  `json:"count"`
	}
	var byStat []StatusCount
	DB.Model(&PlateTrainingSample{}).
		Select("status, count(*) as count").
		Group("status").Find(&byStat)

	type ClassCount struct {
		ClassName string `json:"class_name"`
		Count     int64  `json:"count"`
	}
	// Count per class from approved samples using JSON unnesting (simple approach)
	var approvedSamples []PlateTrainingSample
	DB.Where("status = 'approved'").Select("corrected_text").Find(&approvedSamples)
	classCounts := map[string]int64{}
	for _, s := range approvedSamples {
		text := s.CorrectedText
		if text == "" {
			text = s.RawText
		}
		// Each rune = one char class
		for _, ch := range text {
			key := string(ch)
			if key != " " && key != "-" {
				classCounts[key]++
			}
		}
	}

	var classStats []ClassCount
	for k, v := range classCounts {
		classStats = append(classStats, ClassCount{ClassName: k, Count: v})
	}

	var totalPending int64
	DB.Model(&PlateTrainingSample{}).Where("status = 'pending'").Count(&totalPending)

	return c.JSON(fiber.Map{
		"by_status":    byStat,
		"by_class":     classStats,
		"total_pending": totalPending,
	})
}

// ── Export ZIP ────────────────────────────────────────────────────────────────

// MASTER_CLASSES must match train_char_model.py MASTER_CLASSES order
var masterClasses = []string{
	"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
	"A01", "A02", "A04", "A06", "A07", "A08", "A09", "A10", "A11", "A12",
	"A13", "A14", "A15", "A16", "A17", "A18", "A19", "A20", "A21", "A22",
	"A23", "A24", "A25", "A26", "A27", "A28", "A29", "A30", "A31", "A32",
	"A33", "A34", "A35", "A36", "A37", "A38", "A39", "A40", "A41", "A42",
	"A43", "A44",
	"ACR", "ATG", "AYA", "BKK", "BKN", "BRM", "CBI", "CCO", "CMI", "CNT",
	"CPM", "CPN", "CRI", "CTI", "KBI", "KKN", "KPT", "KRI", "KSN", "LEI",
	"LPG", "LPN", "LRI", "MDH", "MKM", "NAN", "NBI", "NBP", "NKI", "NMA",
	"NPM", "NPT", "NRT", "NSN", "NST", "NWT", "NYK", "PBI", "PCT", "PKN",
	"PKT", "PLG", "PLK", "PNA", "PNB", "PRE", "PRI", "PTE", "PTN", "PYO",
	"RBR", "RET", "RNG", "RYG", "SBR", "SKA", "SKM", "SKN", "SKW", "SNI",
	"SNK", "SPB", "SPK", "SRI", "SRN", "SSK", "STI", "STN", "TAK", "TRG",
	"TRT", "UBN", "UDN", "UTI", "UTT", "YLA", "YST",
}

var classIndex = func() map[string]int {
	m := map[string]int{}
	for i, c := range masterClasses {
		m[c] = i
	}
	return m
}()

type charLabel struct {
	ClassName  string  `json:"class_name"`
	CX         float64 `json:"cx"`
	CY         float64 `json:"cy"`
	BW         float64 `json:"bw"`
	BH         float64 `json:"bh"`
	Confidence float64 `json:"confidence"`
}

func exportTrainingZip(c *fiber.Ctx) error {
	statusFilter := c.Query("status", "approved")
	confMaxStr := c.Query("conf_max", "")
	valSplitStr := c.Query("val_split", "0.1")

	valSplit := 0.1
	if v, err := strconv.ParseFloat(valSplitStr, 64); err == nil && v >= 0 && v <= 0.5 {
		valSplit = v
	}

	query := DB.Model(&PlateTrainingSample{}).Where("image_path != ''")
	if statusFilter != "" {
		query = query.Where("status = ?", statusFilter)
	}
	if confMaxStr != "" {
		if v, err := strconv.ParseFloat(confMaxStr, 64); err == nil {
			query = query.Where("confidence <= ?", v)
		}
	}

	var samples []PlateTrainingSample
	query.Find(&samples)

	if len(samples) == 0 {
		return c.Status(404).JSON(fiber.Map{"error": "no samples found"})
	}

	// Shuffle and split train/val
	rand.Shuffle(len(samples), func(i, j int) { samples[i], samples[j] = samples[j], samples[i] })
	valCount := int(float64(len(samples)) * valSplit)
	if valCount < 1 && len(samples) > 1 {
		valCount = 1
	}
	valSamples := samples[:valCount]
	trainSamples := samples[valCount:]

	buf := new(bytes.Buffer)
	zw := zip.NewWriter(buf)

	writeSplit := func(splitName string, set []PlateTrainingSample) error {
		for i, s := range set {
			stem := fmt.Sprintf("%s_%05d", splitName, i)

			// Fetch image from S3
			if S3Client != nil {
				obj, err := S3Client.GetObject(context.Background(), SnapshotsBucket, s.ImagePath, minio.GetObjectOptions{})
				if err == nil {
					imgBuf := new(bytes.Buffer)
					imgBuf.ReadFrom(obj)
					obj.Close()
					if imgBuf.Len() > 0 {
						fw, _ := zw.Create(fmt.Sprintf("dataset/%s/images/%s.jpg", splitName, stem))
						fw.Write(imgBuf.Bytes())
					}
				}
			}

			// Build YOLO label from char_labels JSON
			labelLines := buildYoloLabel(s)
			if labelLines != "" {
				fw, _ := zw.Create(fmt.Sprintf("dataset/%s/labels/%s.txt", splitName, stem))
				fw.Write([]byte(labelLines))
			}
		}
		return nil
	}

	writeSplit("train", trainSamples)
	writeSplit("valid", valSamples)

	// data.yaml
	classNames := make([]string, len(masterClasses))
	for i, c := range masterClasses {
		classNames[i] = fmt.Sprintf("  - '%s'", c)
	}
	yamlContent := fmt.Sprintf(
		"nc: %d\nnames:\n%s\ntrain: train/images\nval: valid/images\n",
		len(masterClasses), strings.Join(classNames, "\n"),
	)
	fw, _ := zw.Create("dataset/data.yaml")
	fw.Write([]byte(yamlContent))

	// README
	readmeContent := fmt.Sprintf(
		"# Thai License Plate Training Dataset\n\nGenerated: %s\nTrain: %d samples\nValid: %d samples\nClasses: %d\n",
		time.Now().Format(time.RFC3339), len(trainSamples), len(valSamples), len(masterClasses),
	)
	fw2, _ := zw.Create("dataset/README.md")
	fw2.Write([]byte(readmeContent))

	zw.Close()

	c.Set("Content-Type", "application/zip")
	c.Set("Content-Disposition", fmt.Sprintf(`attachment; filename="plate_dataset_%s.zip"`, time.Now().Format("20060102_150405")))
	return c.Send(buf.Bytes())
}

func getExportPreview(c *fiber.Ctx) error {
	statusFilter := c.Query("status", "approved")
	confMaxStr := c.Query("conf_max", "")

	query := DB.Model(&PlateTrainingSample{}).Where("image_path != ''")
	if statusFilter != "" {
		query = query.Where("status = ?", statusFilter)
	}
	if confMaxStr != "" {
		if v, err := strconv.ParseFloat(confMaxStr, 64); err == nil {
			query = query.Where("confidence <= ?", v)
		}
	}

	var total int64
	query.Count(&total)

	return c.JSON(fiber.Map{
		"total":      total,
		"status":     statusFilter,
		"conf_max":   confMaxStr,
	})
}

// buildYoloLabel converts char_labels JSON to YOLO .txt format.
// Uses corrected_text to override class names if available.
func buildYoloLabel(s PlateTrainingSample) string {
	var labels []charLabel
	json.Unmarshal([]byte(s.CharLabels), &labels) //nolint:errcheck

	if len(labels) == 0 {
		return ""
	}

	// If corrected text available, override class_name per position
	corrected := []rune(strings.ReplaceAll(strings.ReplaceAll(s.CorrectedText, " ", ""), "-", ""))
	if len(corrected) == len(labels) {
		for i := range labels {
			labels[i].ClassName = string(corrected[i])
		}
	}

	var lines []string
	for _, lbl := range labels {
		classID, ok := classIndex[lbl.ClassName]
		if !ok {
			continue
		}
		lines = append(lines, fmt.Sprintf("%d %.6f %.6f %.6f %.6f", classID, lbl.CX, lbl.CY, lbl.BW, lbl.BH))
	}
	return strings.Join(lines, "\n")
}
