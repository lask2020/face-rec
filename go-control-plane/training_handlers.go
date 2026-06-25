package main

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/minio/minio-go/v7"

	facerec "github.com/face-rec/go-control-plane/facerec"
)

// ── Thai char ↔ code-name mapping ─────────────────────────────────────────────
// Reverse of engine.py CHAR_LABEL_MAP: Thai unicode / province name → MASTER_CLASSES code.

var thaiToCode = map[string]string{
	// Thai consonants
	"ก": "A01", "ข": "A02", "ค": "A04", "ฆ": "A06",
	"ง": "A07", "จ": "A08", "ฉ": "A09", "ช": "A10",
	"ซ": "A11", "ฌ": "A12", "ญ": "A13", "ฎ": "A14",
	"ฏ": "A15", "ฐ": "A16", "ฑ": "A17", "ฒ": "A18",
	"ณ": "A19", "ด": "A20", "ต": "A21", "ถ": "A22",
	"ท": "A23", "ธ": "A24", "น": "A25", "บ": "A26",
	"ป": "A27", "ผ": "A28", "ฝ": "A29", "พ": "A30",
	"ฟ": "A31", "ภ": "A32", "ม": "A33", "ย": "A34",
	"ร": "A35", "ล": "A36", "ว": "A37", "ศ": "A38",
	"ษ": "A39", "ส": "A40", "ห": "A41", "ฬ": "A42",
	"อ": "A43", "ฮ": "A44",
	// Province Thai names → codes
	"อ่างทอง": "ATG", "อยุธยา": "AYA",
	"กรุงเทพ": "BKK", "บึงกาฬ": "BKN", "บุรีรัมย์": "BRM",
	"ชลบุรี": "CBI", "ฉะเชิงเทรา": "CCO", "เชียงใหม่": "CMI",
	"ชัยนาท": "CNT", "ชัยภูมิ": "CPM", "ชุมพร": "CPN",
	"เชียงราย": "CRI", "จันทบุรี": "CTI",
	"กระบี่": "KBI", "ขอนแก่น": "KKN", "กาฬสินธุ์": "KSN",
	"กาญจนบุรี": "KRI", "กำแพงเพชร": "KPT",
	"เลย": "LEI", "ลำปาง": "LPG", "ลำพูน": "LPN", "ลพบุรี": "LRI",
	"มหาสารคาม": "MDH", "มุกดาหาร": "MKM",
	"น่าน": "NAN", "หนองบัวลำภู": "NBI", "นนทบุรี": "NBP",
	"หนองคาย": "NKI", "นครราชสีมา": "NMA", "นครปฐม": "NPM",
	"นครพนม": "NPT", "นราธิวาส": "NRT", "นครสวรรค์": "NSN",
	"นครศรีธรรมราช": "NST", "นครนายก": "NWT",
	"ปราจีนบุรี": "PBI", "ประจวบคีรีขันธ์": "PCT", "ประจวบฯ": "PKN",
	"ภูเก็ต": "PKT", "พัทลุง": "PLG", "พิษณุโลก": "PLK",
	"พังงา": "PNA", "เพชรบูรณ์": "PNB", "เพชรบุรี": "PTE",
	"แพร่": "PRE", "ปทุมธานี": "PRI", "ปัตตานี": "PTN",
	"พะเยา": "PYO",
	"ราชบุรี": "RBR", "ร้อยเอ็ด": "RET", "ระนอง": "RNG", "ระยอง": "RYG",
	"สระบุรี": "SBR", "สงขลา": "SKA", "สกลนคร": "SKM",
	"สมุทรสาคร": "SKN", "สมุทรสงคราม": "SKW",
	"สิงห์บุรี": "SNI", "สกลนคร2": "SNK",
	"สุพรรณบุรี": "SPB", "สมุทรปราการ": "SPK",
	"สุรินทร์": "SRI", "สุราษฎร์ธานี": "SRN",
	"ศรีสะเกษ": "SSK", "สตูล": "STI", "สุโขทัย": "STN",
	"ตาก": "TAK", "ตรัง": "TRG", "ตราด": "TRT",
	"อุบลราชธานี": "UBN", "อุดรธานี": "UDN",
	"อุทัยธานี": "UTI", "อุตรดิตถ์": "UTT",
	"ยะลา": "YLA", "ยโสธร": "YST",
}

// normalizeToCode converts a Thai unicode char or province name to its MASTER_CLASSES code.
// Digits and already-valid codes pass through unchanged.
func normalizeToCode(name string) string {
	if _, ok := classIndex[name]; ok {
		return name // already a valid code or digit
	}
	if code, ok := thaiToCode[name]; ok {
		return code
	}
	return name // unknown — keep as-is and let classIndex miss it
}

// ── Dedup cache ───────────────────────────────────────────────────────────────
// Prevents saving duplicate training samples for the same plate on the same camera
// within a short time window.

const trainingDedupWindow = 5 * time.Minute

var (
	trainingDedupCache   = make(map[string]time.Time)
	trainingDedupCacheMu sync.Mutex
)

func trainingDedupKey(cameraID uint, rawText string) string {
	return fmt.Sprintf("%d|%s", cameraID, rawText)
}

func claimTrainingSlot(cameraID uint, rawText string) bool {
	key := trainingDedupKey(cameraID, rawText)
	now := time.Now()
	trainingDedupCacheMu.Lock()
	defer trainingDedupCacheMu.Unlock()
	if exp, ok := trainingDedupCache[key]; ok && now.Before(exp) {
		return false
	}
	trainingDedupCache[key] = now.Add(trainingDedupWindow)
	// Opportunistically evict expired entries to keep map small
	for k, exp := range trainingDedupCache {
		if now.After(exp) {
			delete(trainingDedupCache, k)
		}
	}
	return true
}

// ── S3 Save ───────────────────────────────────────────────────────────────────

func saveTrainingFrames(ctx context.Context, frames []*facerec.PlateTrainingFrame, task PendingTask) {
	if len(frames) == 0 {
		return
	}

	var cam Camera
	DB.First(&cam, task.CameraID)

	// Use raw_text of first frame for dedup key (all frames in a track share the same plate)
	rawText := frames[0].RawText
	if !claimTrainingSlot(task.CameraID, rawText) {
		log.Printf("[Training] Skipping duplicate training sample cam=%d raw=%q", task.CameraID, rawText)
		return
	}

	for idx, frame := range frames {
		if len(frame.CropJpeg) == 0 {
			continue
		}

		sample := PlateTrainingSample{
			CameraID:   task.CameraID,
			CameraName: cam.Name,
			TrackID:    frame.TrackId,
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
	search := c.Query("search", "")

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
	if search != "" {
		like := "%" + strings.ToLower(search) + "%"
		query = query.Where("LOWER(raw_text) LIKE ? OR LOWER(corrected_text) LIKE ? OR LOWER(camera_name) LIKE ?", like, like, like)
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

func clearTrainingSamples(c *fiber.Ctx) error {
	status := c.Query("status", "") // empty = all statuses

	query := DB.Model(&PlateTrainingSample{})
	if status != "" {
		query = query.Where("status = ?", status)
	}

	// Collect S3 keys before deleting rows
	var samples []PlateTrainingSample
	query.Find(&samples)

	if S3Client != nil {
		for _, s := range samples {
			if s.ImagePath != "" {
				if err := S3Client.RemoveObject(c.Context(), SnapshotsBucket, s.ImagePath, minio.RemoveObjectOptions{}); err != nil {
					log.Printf("[Training] S3 delete failed for %s: %v", s.ImagePath, err)
				}
			}
		}
	}

	del := DB.Where("1 = 1")
	if status != "" {
		del = DB.Where("status = ?", status)
	}
	result := del.Delete(&PlateTrainingSample{})
	if result.Error != nil {
		return c.Status(500).JSON(fiber.Map{"error": result.Error.Error()})
	}

	log.Printf("[Training] Cleared %d samples (status=%q)", result.RowsAffected, status)
	return c.JSON(fiber.Map{"deleted": result.RowsAffected})
}

func updateTrainingTrack(c *fiber.Ctx) error {
	trackID := c.Params("track_id")
	if trackID == "" {
		return c.Status(400).JSON(fiber.Map{"error": "track_id required"})
	}

	var body struct {
		Status        string `json:"status"`
		CorrectedText string `json:"corrected_text"`
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
	if len(updates) == 0 {
		return c.Status(400).JSON(fiber.Map{"error": "nothing to update"})
	}

	result := DB.Model(&PlateTrainingSample{}).Where("track_id = ?", trackID).Updates(updates)
	return c.JSON(fiber.Map{"updated": result.RowsAffected, "track_id": trackID})
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

	// Count class distribution from char_labels JSON across all non-rejected samples.
	// Using char_labels (per-character detections) is more accurate than parsing raw_text
	// character-by-character, which breaks province codes like "BKK" → "B","K","K".
	var samples []PlateTrainingSample
	DB.Where("status != 'rejected' AND char_labels != '' AND char_labels != '[]'").
		Select("char_labels").Find(&samples)

	type charLabelEntry struct {
		ClassName string `json:"class_name"`
	}
	classCounts := map[string]int64{}
	for _, s := range samples {
		var labels []charLabelEntry
		if err := json.Unmarshal([]byte(s.CharLabels), &labels); err != nil {
			continue
		}
		for _, lbl := range labels {
			code := normalizeToCode(lbl.ClassName)
			if _, ok := classIndex[code]; ok {
				classCounts[code]++
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
		"by_status":     byStat,
		"by_class":      classStats,
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

	query := DB.Model(&PlateTrainingSample{})
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

	writeSplit := func(splitName string, set []PlateTrainingSample) {
		for i, s := range set {
			labelLines := buildYoloLabel(s)
			if labelLines == "" {
				continue // skip samples with no usable labels
			}
			stem := fmt.Sprintf("%s_%05d", splitName, i)

			// Fetch image from S3 using raw ImagePath (S3 key, not the URL)
			if S3Client != nil && s.ImagePath != "" {
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

			fw, _ := zw.Create(fmt.Sprintf("dataset/%s/labels/%s.txt", splitName, stem))
			fw.Write([]byte(labelLines))
		}
	}

	writeSplit("train", trainSamples)
	writeSplit("valid", valSamples)

	// data.yaml
	classNames := make([]string, len(masterClasses))
	for i, cls := range masterClasses {
		classNames[i] = fmt.Sprintf("  - '%s'", cls)
	}
	yamlContent := fmt.Sprintf(
		"nc: %d\nnames:\n%s\ntrain: train/images\nval: valid/images\n",
		len(masterClasses), strings.Join(classNames, "\n"),
	)
	fw, _ := zw.Create("dataset/data.yaml")
	fw.Write([]byte(yamlContent))

	withImage := 0
	for _, s := range samples {
		if s.ImagePath != "" {
			withImage++
		}
	}
	readmeContent := fmt.Sprintf(
		"# Thai License Plate Training Dataset\n\nGenerated: %s\nTotal: %d samples (%d with images)\nTrain: %d  Valid: %d\nClasses: %d\n",
		time.Now().Format(time.RFC3339), len(samples), withImage, len(trainSamples), len(valSamples), len(masterClasses),
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

	query := DB.Model(&PlateTrainingSample{})
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
		"total":    total,
		"status":   statusFilter,
		"conf_max": confMaxStr,
	})
}

// ── Finetune job ──────────────────────────────────────────────────────────────

type finetuneJobState struct {
	mu        sync.Mutex
	Status    string    `json:"status"`    // idle | running | done | error
	StartedAt time.Time `json:"started_at"`
	Epoch     int       `json:"epoch"`
	Epochs    int       `json:"epochs"`
	Log       []string  `json:"log"`
	Error     string    `json:"error"`
}

var finetuneJob = &finetuneJobState{Status: "idle"}

const maxFinetuneLog = 200

func (j *finetuneJobState) appendLog(line string) {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.Log = append(j.Log, line)
	if len(j.Log) > maxFinetuneLog {
		j.Log = j.Log[len(j.Log)-maxFinetuneLog:]
	}
}

func (j *finetuneJobState) snapshot() map[string]interface{} {
	j.mu.Lock()
	defer j.mu.Unlock()
	logCopy := make([]string, len(j.Log))
	copy(logCopy, j.Log)
	var startedAt interface{}
	if j.Status != "idle" {
		startedAt = j.StartedAt.Format(time.RFC3339)
	}
	return map[string]interface{}{
		"status":     j.Status,
		"started_at": startedAt,
		"epoch":      j.Epoch,
		"epochs":     j.Epochs,
		"log":        logCopy,
		"error":      j.Error,
	}
}

func (j *finetuneJobState) setError(msg string) {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.Status = "error"
	j.Error = msg
}

// zipDirectory creates a zip archive of src directory at dst path.
func zipDirectory(src, dst string) error {
	f, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer f.Close()
	w := zip.NewWriter(f)
	defer w.Close()
	return filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return err
		}
		rel, _ := filepath.Rel(src, path)
		zf, err := w.Create(rel)
		if err != nil {
			return err
		}
		in, err := os.Open(path)
		if err != nil {
			return err
		}
		defer in.Close()
		_, err = io.Copy(zf, in)
		return err
	})
}

// BroadcastFinetuneTask sends a start_finetune task to all connected workers.
// Returns the number of workers that received it.
func BroadcastFinetuneTask(task *facerec.FrameTask) int {
	activeWorkersMu.Lock()
	ws := make([]*AIWorkerSession, len(activeWorkers))
	copy(ws, activeWorkers)
	activeWorkersMu.Unlock()

	delivered := 0
	for _, w := range ws {
		if w.sendBlocking(task, 10*time.Second) {
			delivered++
		}
	}
	return delivered
}

// SendFinetuneToWorker sends a finetune task to a specific worker by ID.
// Returns true if the worker was found and task was delivered.
func SendFinetuneToWorker(workerID string, task *facerec.FrameTask) bool {
	activeWorkersMu.Lock()
	ws := make([]*AIWorkerSession, len(activeWorkers))
	copy(ws, activeWorkers)
	activeWorkersMu.Unlock()

	for _, w := range ws {
		if w.id == workerID {
			return w.sendBlocking(task, 10*time.Second)
		}
	}
	return false
}

// exportDatasetToDir writes the approved YOLO dataset to a temp directory and
// returns the path to data.yaml (or an error).
func exportDatasetToDir(dir string) (string, error) {
	if S3Client == nil {
		return "", fmt.Errorf("S3 not configured — cannot fetch training images")
	}
	var samples []PlateTrainingSample
	DB.Where("status = 'approved'").Find(&samples)
	if len(samples) == 0 {
		return "", fmt.Errorf("no approved samples found")
	}

	rand.Shuffle(len(samples), func(i, j int) { samples[i], samples[j] = samples[j], samples[i] })
	valCount := int(float64(len(samples)) * 0.1)
	if valCount < 1 && len(samples) > 1 {
		valCount = 1
	}

	splits := []struct {
		name    string
		samples []PlateTrainingSample
	}{
		{"train", samples[valCount:]},
		{"valid", samples[:valCount]},
	}

	for _, sp := range splits {
		for _, sub := range []string{"images", "labels"} {
			if err := os.MkdirAll(filepath.Join(dir, sp.name, sub), 0755); err != nil {
				return "", err
			}
		}
	}

	for _, sp := range splits {
		for i, s := range sp.samples {
			labelLines := buildYoloLabel(s)
			if labelLines == "" {
				continue
			}
			if S3Client == nil || s.ImagePath == "" {
				continue
			}

			// Fetch image first — only write label if image succeeds
			obj, err := S3Client.GetObject(context.Background(), SnapshotsBucket, s.ImagePath, minio.GetObjectOptions{})
			if err != nil {
				continue
			}
			imgData, _ := io.ReadAll(obj)
			obj.Close()
			if len(imgData) == 0 {
				continue
			}

			stem := fmt.Sprintf("%s_%05d", sp.name, i)
			imgPath := filepath.Join(dir, sp.name, "images", stem+".jpg")
			if err := os.WriteFile(imgPath, imgData, 0644); err != nil {
				return "", err
			}
			lblPath := filepath.Join(dir, sp.name, "labels", stem+".txt")
			if err := os.WriteFile(lblPath, []byte(labelLines), 0644); err != nil {
				return "", err
			}
		}
	}

	// data.yaml with absolute paths
	classNames := make([]string, len(masterClasses))
	for i, cls := range masterClasses {
		classNames[i] = fmt.Sprintf("  - '%s'", cls)
	}
	yamlContent := fmt.Sprintf(
		"nc: %d\nnames:\n%s\ntrain: %s\nval: %s\n",
		len(masterClasses),
		strings.Join(classNames, "\n"),
		filepath.Join(dir, "train", "images"),
		filepath.Join(dir, "valid", "images"),
	)
	yamlPath := filepath.Join(dir, "data.yaml")
	if err := os.WriteFile(yamlPath, []byte(yamlContent), 0644); err != nil {
		return "", err
	}
	return yamlPath, nil
}

func resolveModelsDir() string {
	if root := os.Getenv("FACE_DATA_ROOT"); root != "" {
		return filepath.Join(root, "models")
	}
	for _, candidate := range []string{"/app/data/models", "data/models", "backend/data/models"} {
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
	}
	return "data/models"
}

func startFinetune(c *fiber.Ctx) error {
	var body struct {
		Epochs   int    `json:"epochs"`
		WorkerID string `json:"worker_id"`
	}
	_ = c.BodyParser(&body)

	finetuneJob.mu.Lock()
	if finetuneJob.Status == "running" {
		finetuneJob.mu.Unlock()
		return c.Status(409).JSON(fiber.Map{"error": "finetune job already running"})
	}
	finetuneJob.Status = "running"
	finetuneJob.StartedAt = time.Now()
	finetuneJob.Epoch = 0
	finetuneJob.Epochs = 0
	finetuneJob.Log = nil
	finetuneJob.Error = ""
	finetuneJob.mu.Unlock()

	go func() {
		defer func() {
			if r := recover(); r != nil {
				finetuneJob.mu.Lock()
				finetuneJob.Status = "error"
				finetuneJob.Error = fmt.Sprintf("panic: %v", r)
				finetuneJob.mu.Unlock()
			}
		}()

		// 1. Export approved CCTV samples to temp dir
		tmpDir, err := os.MkdirTemp("", "finetune_*")
		if err != nil {
			finetuneJob.setError("failed to create temp dir: " + err.Error())
			return
		}
		defer os.RemoveAll(tmpDir)

		finetuneJob.appendLog("Exporting approved samples...")
		if _, err := exportDatasetToDir(tmpDir); err != nil {
			finetuneJob.setError("export failed: " + err.Error())
			return
		}
		finetuneJob.appendLog("Export done. Zipping dataset...")

		// 2. Zip the exported dataset
		zipPath := tmpDir + ".zip"
		defer os.Remove(zipPath)
		if err := zipDirectory(tmpDir, zipPath); err != nil {
			finetuneJob.setError("zip failed: " + err.Error())
			return
		}

		// 3. Upload zip to S3
		s3Key := fmt.Sprintf("finetune_dataset_%d.zip", time.Now().UnixMilli())
		finetuneJob.appendLog("Uploading dataset to S3...")
		f, err := os.Open(zipPath)
		if err != nil {
			finetuneJob.setError("open zip: " + err.Error())
			return
		}
		stat, _ := f.Stat()
		_, err = S3Client.PutObject(context.Background(), SnapshotsBucket, s3Key, f, stat.Size(),
			minio.PutObjectOptions{ContentType: "application/zip"})
		f.Close()
		if err != nil {
			finetuneJob.setError("S3 upload failed: " + err.Error())
			return
		}
		finetuneJob.appendLog("Dataset uploaded. Sending to AI worker...")

		// 4. Broadcast start_finetune to workers via gRPC
		// Priority: request body > env var > default 30
		epochs := int32(30)
		if body.Epochs > 0 {
			epochs = int32(body.Epochs)
		} else if epochsStr := os.Getenv("FINETUNE_EPOCHS"); epochsStr != "" {
			if n, err := strconv.Atoi(epochsStr); err == nil {
				epochs = int32(n)
			}
		}
		task := &facerec.FrameTask{
			StartFinetune:        true,
			FinetuneDatasetS3Key: s3Key,
			FinetuneEpochs:       epochs,
			RoboflowApiKey:       getSettingValue("roboflow_api_key"),
		}
		var delivered int
		if body.WorkerID != "" {
			if SendFinetuneToWorker(body.WorkerID, task) {
				delivered = 1
			} else {
				finetuneJob.setError(fmt.Sprintf("worker %s not found or not connected", body.WorkerID))
				S3Client.RemoveObject(context.Background(), SnapshotsBucket, s3Key, minio.RemoveObjectOptions{})
				return
			}
		} else {
			// No specific worker requested — pick the first available worker.
			// Broadcasting to all workers would cause each to train independently
			// and race on activateVersion, so we always train on exactly one worker.
			w := getNextWorker()
			if w == nil {
				finetuneJob.setError("no AI workers connected — cannot start training")
				S3Client.RemoveObject(context.Background(), SnapshotsBucket, s3Key, minio.RemoveObjectOptions{})
				return
			}
			if w.sendBlocking(task, 10*time.Second) {
				delivered = 1
				finetuneJob.appendLog(fmt.Sprintf("No worker specified — auto-selected worker %s", w.id))
			} else {
				finetuneJob.setError(fmt.Sprintf("worker %s queue full — cannot start training", w.id))
				S3Client.RemoveObject(context.Background(), SnapshotsBucket, s3Key, minio.RemoveObjectOptions{})
				return
			}
		}
		finetuneJob.appendLog(fmt.Sprintf("Training started on %d worker(s)", delivered))
		// Status stays "running" — worker will send FinetuneProgress back via gRPC
	}()

	return c.JSON(fiber.Map{"status": "started"})
}

func getFinetuneStatus(c *fiber.Ctx) error {
	return c.JSON(finetuneJob.snapshot())
}

func stopFinetune(c *fiber.Ctx) error {
	finetuneJob.mu.Lock()
	if finetuneJob.Status != "running" {
		finetuneJob.mu.Unlock()
		return c.Status(409).JSON(fiber.Map{"error": "no finetune job running"})
	}
	finetuneJob.mu.Unlock()

	task := &facerec.FrameTask{StopFinetune: true}
	delivered := BroadcastFinetuneTask(task)
	if delivered == 0 {
		return c.Status(503).JSON(fiber.Map{"error": "no AI workers connected"})
	}
	finetuneJob.appendLog("Stop signal sent to worker")
	return c.JSON(fiber.Map{"status": "stop_sent"})
}

// buildYoloLabel converts char_labels JSON to YOLO .txt format.
// Handles Thai unicode → code name conversion, and applies corrected_text override.
func buildYoloLabel(s PlateTrainingSample) string {
	var labels []charLabel
	json.Unmarshal([]byte(s.CharLabels), &labels) //nolint:errcheck
	if len(labels) == 0 {
		return ""
	}

	// Apply corrected_text override: strip spaces/dashes, then map per position
	corrected := []rune(strings.ReplaceAll(strings.ReplaceAll(s.CorrectedText, " ", ""), "-", ""))
	if len(corrected) == len(labels) {
		for i := range labels {
			labels[i].ClassName = string(corrected[i])
		}
	}

	var lines []string
	for _, lbl := range labels {
		// Normalize Thai unicode → MASTER_CLASSES code before lookup
		code := normalizeToCode(lbl.ClassName)
		classID, ok := classIndex[code]
		if !ok {
			log.Printf("[Training] Unknown class %q (normalized: %q) — skipping", lbl.ClassName, code)
			continue
		}
		lines = append(lines, fmt.Sprintf("%d %.6f %.6f %.6f %.6f", classID, lbl.CX, lbl.CY, lbl.BW, lbl.BH))
	}
	return strings.Join(lines, "\n")
}

// ── Model Version Management ──────────────────────────────────────────────────

type ModelVersionMeta struct {
	Version   string `json:"version"`
	Label     string `json:"label"`
	TrainedAt string `json:"trained_at"`
	Samples   int    `json:"samples"`
	Epochs    int    `json:"epochs"`
	BaseModel string `json:"base_model"`
	HasOnnx   bool   `json:"has_onnx"`
	Active    bool   `json:"active"`
}

func listModelVersions(c *fiber.Ctx) error {
	versionsDir := filepath.Join(resolveModelsDir(), "versions")
	entries, err := os.ReadDir(versionsDir)
	if err != nil {
		if os.IsNotExist(err) {
			return c.JSON(fiber.Map{"versions": []interface{}{}})
		}
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	activeVersion := readActiveVersion()

	var versions []ModelVersionMeta
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		metaPath := filepath.Join(versionsDir, e.Name(), "meta.json")
		data, err := os.ReadFile(metaPath)
		if err != nil {
			continue
		}
		var meta ModelVersionMeta
		if err := json.Unmarshal(data, &meta); err != nil {
			continue
		}
		onnxPath := filepath.Join(versionsDir, e.Name(), "thai_char_yolo26s.onnx")
		meta.HasOnnx = fileExists(onnxPath)
		meta.Active = meta.Version == activeVersion
		versions = append(versions, meta)
	}

	// newest first
	for i, j := 0, len(versions)-1; i < j; i, j = i+1, j-1 {
		versions[i], versions[j] = versions[j], versions[i]
	}

	if versions == nil {
		versions = []ModelVersionMeta{}
	}
	return c.JSON(fiber.Map{"versions": versions, "active": activeVersion})
}

// activateVersion copies a stored version's model files into the active model
// dir, marks it active, then asynchronously pushes to S3 and reloads workers.
// Shared by the deploy endpoint and the gRPC finetune "done" handler.
func activateVersion(version string) error {
	modelsDir := resolveModelsDir()
	versionDir := filepath.Join(modelsDir, "versions", version)
	if !fileExists(versionDir) {
		return fmt.Errorf("version not found: %s", version)
	}

	ptSrc := filepath.Join(versionDir, "thai_char_yolo26s.pt")
	ptDst := filepath.Join(modelsDir, "thai_char_yolo26s.pt")
	if fileExists(ptSrc) {
		if err := copyFile(ptSrc, ptDst); err != nil {
			return fmt.Errorf("copy .pt failed: %w", err)
		}
	}

	onnxSrc := filepath.Join(versionDir, "thai_char_yolo26s.onnx")
	onnxDst := filepath.Join(modelsDir, "thai_char_yolo26s.onnx")
	if fileExists(onnxSrc) {
		if err := copyFile(onnxSrc, onnxDst); err != nil {
			return fmt.Errorf("copy .onnx failed: %w", err)
		}
	}

	writeActiveVersion(version)

	// Push to S3 first, THEN broadcast reload — otherwise workers would pull the
	// stale/partial model from S3 before the upload finishes. Run async so the
	// caller returns immediately; the actual reload happens in background.
	go func() {
		pushModelsToS3(version)
		n := BroadcastReloadModels()
		log.Printf("[ModelActivate] version %s pushed to S3, reload delivered to %d worker(s)", version, n)
	}()

	return nil
}

// writeVersionMeta writes meta.json for a version dir (control-plane authoritative).
func writeVersionMeta(version string, epochs int) {
	var samples int64
	DB.Model(&PlateTrainingSample{}).Where("status = 'approved'").Count(&samples)
	meta := ModelVersionMeta{
		Version:   version,
		TrainedAt: time.Now().UTC().Format(time.RFC3339),
		Samples:   int(samples),
		Epochs:    epochs,
		BaseModel: "thai_char_yolo26s.pt",
	}
	dir := filepath.Join(resolveModelsDir(), "versions", version)
	os.MkdirAll(dir, 0o755) //nolint:errcheck
	b, _ := json.Marshal(meta)
	os.WriteFile(filepath.Join(dir, "meta.json"), b, 0o644) //nolint:errcheck
}

func deployModelVersion(c *fiber.Ctx) error {
	version := c.Params("version")
	if version == "" {
		return c.Status(400).JSON(fiber.Map{"error": "version required"})
	}

	if err := activateVersion(version); err != nil {
		if strings.Contains(err.Error(), "not found") {
			return c.Status(404).JSON(fiber.Map{"error": err.Error()})
		}
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	log.Printf("[ModelDeploy] Deploying version %s — pushing to S3 then reloading workers", version)
	return c.JSON(fiber.Map{"deployed": version, "status": "deploying"})
}

// snapshotModelVersion copies the CURRENTLY ACTIVE char model into a new version
// dir so it can always be restored later. Essential as a rollback point before
// experimenting with new fine-tunes (the hand-trained baseline would otherwise be
// overwritten with no way back).
func snapshotModelVersion(c *fiber.Ctx) error {
	var body struct {
		Label string `json:"label"`
	}
	_ = c.BodyParser(&body)

	modelsDir := resolveModelsDir()
	ptSrc := filepath.Join(modelsDir, "thai_char_yolo26s.pt")
	onnxSrc := filepath.Join(modelsDir, "thai_char_yolo26s.onnx")

	// The active model's canonical store is S3 — the control plane's local
	// data/models/ is only a staging dir and is often empty (the model physically
	// lives on the GPU worker). Pull the active files from S3 so a rollback
	// snapshot can be taken even when nothing is on local disk.
	if !fileExists(ptSrc) {
		if err := fetchModelFromS3("thai_char_yolo26s.pt", ptSrc); err != nil {
			log.Printf("[ModelSnapshot] could not fetch thai_char_yolo26s.pt from S3: %v", err)
		}
	}
	if !fileExists(onnxSrc) {
		if err := fetchModelFromS3("thai_char_yolo26s.onnx", onnxSrc); err != nil {
			log.Printf("[ModelSnapshot] could not fetch thai_char_yolo26s.onnx from S3: %v", err)
		}
	}

	if !fileExists(ptSrc) && !fileExists(onnxSrc) {
		return c.Status(404).JSON(fiber.Map{"error": "no active char model found on disk or in S3 to snapshot"})
	}

	version := time.Now().Format("20060102_150405")
	destDir := filepath.Join(modelsDir, "versions", version)
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	if fileExists(ptSrc) {
		if err := copyFile(ptSrc, filepath.Join(destDir, "thai_char_yolo26s.pt")); err != nil {
			return c.Status(500).JSON(fiber.Map{"error": "copy .pt failed: " + err.Error()})
		}
	}
	if fileExists(onnxSrc) {
		if err := copyFile(onnxSrc, filepath.Join(destDir, "thai_char_yolo26s.onnx")); err != nil {
			return c.Status(500).JSON(fiber.Map{"error": "copy .onnx failed: " + err.Error()})
		}
	}

	label := strings.TrimSpace(body.Label)
	if label == "" {
		label = "snapshot"
	}
	meta := ModelVersionMeta{
		Version:   version,
		Label:     label,
		TrainedAt: time.Now().UTC().Format(time.RFC3339),
		BaseModel: "thai_char_yolo26s.pt (snapshot of active)",
	}
	b, _ := json.Marshal(meta)
	os.WriteFile(filepath.Join(destDir, "meta.json"), b, 0o644) //nolint:errcheck

	log.Printf("[ModelSnapshot] Saved active model as version %s (label=%q)", version, label)
	return c.JSON(meta)
}

// renameModelVersion updates a version's human-readable label.
func renameModelVersion(c *fiber.Ctx) error {
	version := c.Params("version")
	if !validVersionName(version) {
		return c.Status(400).JSON(fiber.Map{"error": "invalid version"})
	}
	var body struct {
		Label string `json:"label"`
	}
	if err := c.BodyParser(&body); err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid body"})
	}
	metaPath := filepath.Join(resolveModelsDir(), "versions", version, "meta.json")
	data, err := os.ReadFile(metaPath)
	if err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "version not found"})
	}
	var meta ModelVersionMeta
	if err := json.Unmarshal(data, &meta); err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}
	meta.Label = strings.TrimSpace(body.Label)
	b, _ := json.Marshal(meta)
	os.WriteFile(metaPath, b, 0o644) //nolint:errcheck
	return c.JSON(meta)
}

// deleteModelVersion removes a stored version. The active version cannot be deleted.
func deleteModelVersion(c *fiber.Ctx) error {
	version := c.Params("version")
	if !validVersionName(version) {
		return c.Status(400).JSON(fiber.Map{"error": "invalid version"})
	}
	if version == readActiveVersion() {
		return c.Status(409).JSON(fiber.Map{"error": "cannot delete the active version — deploy another version first"})
	}
	dir := filepath.Join(resolveModelsDir(), "versions", version)
	if !fileExists(dir) {
		return c.Status(404).JSON(fiber.Map{"error": "version not found"})
	}
	if err := os.RemoveAll(dir); err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}
	log.Printf("[ModelVersion] Deleted version %s", version)
	return c.JSON(fiber.Map{"deleted": version})
}

// ── helpers ───────────────────────────────────────────────────────────────────

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, in)
	return err
}

func activeVersionPath() string {
	return filepath.Join(resolveModelsDir(), "active_version.txt")
}

func readActiveVersion() string {
	data, err := os.ReadFile(activeVersionPath())
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

func writeActiveVersion(version string) {
	os.WriteFile(activeVersionPath(), []byte(version), 0644) //nolint:errcheck
}
