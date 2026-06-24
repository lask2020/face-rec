package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/minio/minio-go/v7"
)

func bytesReader(b []byte) *bytes.Reader { return bytes.NewReader(b) }

// modelFileNames lists all model files that are managed in S3.
var modelFileNames = []string{
	"thai_char_yolo26s.onnx",
	"thai_char_yolo26s.pt",
	"thai_plate_yolo11n.onnx",
	"thai_plate_yolo11n.pt",
}

// ── Internal helper ───────────────────────────────────────────────────────────

// pushModelsToS3 uploads the active local model files to S3 ModelsBucket.
// Called automatically after deployModelVersion.
func pushModelsToS3(version string) {
	if S3Client == nil {
		log.Println("[ModelS3] S3 not configured — skipping push")
		return
	}

	modelsDir := resolveModelsDir()
	ctx := context.Background()

	for _, name := range modelFileNames {
		localPath := filepath.Join(modelsDir, name)
		f, err := os.Open(localPath)
		if err != nil {
			log.Printf("[ModelS3] %s not found locally — skipping", name)
			continue
		}
		stat, _ := f.Stat()
		_, err = S3Client.PutObject(ctx, ModelsBucket, name, f, stat.Size(),
			minio.PutObjectOptions{ContentType: "application/octet-stream"})
		f.Close()
		if err != nil {
			log.Printf("[ModelS3] upload %s failed: %v", name, err)
		} else {
			log.Printf("[ModelS3] uploaded %s (%d bytes)", name, stat.Size())
		}
	}

	// Write meta.json so workers know the active version
	meta := map[string]interface{}{
		"version":    version,
		"updated_at": time.Now().UTC().Format(time.RFC3339),
	}
	metaBytes, _ := json.Marshal(meta)
	_, err := S3Client.PutObject(ctx, ModelsBucket, "meta.json",
		bytesReader(metaBytes), int64(len(metaBytes)),
		minio.PutObjectOptions{ContentType: "application/json"})
	if err != nil {
		log.Printf("[ModelS3] upload meta.json failed: %v", err)
	}
	log.Printf("[ModelS3] Push complete for version %s", version)
}

// ── REST Handlers ─────────────────────────────────────────────────────────────

// GET /api/models/manifest — returns list of model files stored in S3 with sizes/dates.
// AI workers call this on startup to decide what to download.
func getModelsManifest(c *fiber.Ctx) error {
	if S3Client == nil {
		return c.Status(503).JSON(fiber.Map{"error": "S3 not configured"})
	}

	type FileInfo struct {
		Name      string `json:"name"`
		Size      int64  `json:"size"`
		UpdatedAt string `json:"updated_at"`
	}

	ctx := context.Background()
	var files []FileInfo

	for _, name := range append(modelFileNames, "meta.json") {
		info, err := S3Client.StatObject(ctx, ModelsBucket, name, minio.StatObjectOptions{})
		if err != nil {
			continue
		}
		files = append(files, FileInfo{
			Name:      name,
			Size:      info.Size,
			UpdatedAt: info.LastModified.UTC().Format(time.RFC3339),
		})
	}

	if files == nil {
		files = []FileInfo{}
	}
	return c.JSON(fiber.Map{"files": files, "bucket": ModelsBucket})
}

// GET /api/models/download/:filename — streams a model file from S3 to the client.
func downloadModelFile(c *fiber.Ctx) error {
	name := c.Params("filename")
	if name == "" {
		return c.Status(400).JSON(fiber.Map{"error": "filename required"})
	}

	// Only allow known model files + meta.json to prevent path traversal
	allowed := false
	for _, n := range append(modelFileNames, "meta.json") {
		if n == name {
			allowed = true
			break
		}
	}
	if !allowed {
		return c.Status(403).JSON(fiber.Map{"error": "file not allowed"})
	}

	if S3Client == nil {
		return c.Status(503).JSON(fiber.Map{"error": "S3 not configured"})
	}

	ctx := context.Background()
	obj, err := S3Client.GetObject(ctx, ModelsBucket, name, minio.GetObjectOptions{})
	if err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "not found"})
	}
	defer obj.Close()

	info, err := obj.Stat()
	if err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "not found"})
	}

	c.Set("Content-Type", "application/octet-stream")
	c.Set("Content-Disposition", `attachment; filename="`+name+`"`)
	c.Set("Content-Length", strconv.FormatInt(info.Size, 10))
	_, err = io.Copy(c.Response().BodyWriter(), obj)
	return err
}

// POST /api/models/push — manually push current local active models to S3.
// Useful for bootstrapping: run once after placing initial models on disk.
func pushModelsHandler(c *fiber.Ctx) error {
	activeVersion := readActiveVersion()
	if activeVersion == "" {
		activeVersion = "manual_" + time.Now().Format("20060102_150405")
	}
	go pushModelsToS3(activeVersion)
	return c.JSON(fiber.Map{"status": "push started", "version": activeVersion})
}
