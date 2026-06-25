package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/minio/minio-go/v7"
)

func bytesReader(b []byte) *bytes.Reader { return bytes.NewReader(b) }

// fetchModelFromS3 downloads a single model file from S3 ModelsBucket into dest.
// Used when the control plane needs a model that lives only in S3 (the canonical
// store) and not on its local staging disk — e.g. taking a rollback snapshot on
// a control plane that has never run a fine-tune locally. Returns an error if S3
// is unconfigured or the object does not exist.
func fetchModelFromS3(name, dest string) error {
	if S3Client == nil {
		return fmt.Errorf("S3 not configured")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	obj, err := S3Client.GetObject(ctx, ModelsBucket, name, minio.GetObjectOptions{})
	if err != nil {
		return err
	}
	defer obj.Close()
	// GetObject is lazy — Stat surfaces a missing-object error before we write.
	if _, err := obj.Stat(); err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}
	tmp := dest + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, obj); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	f.Close()
	return os.Rename(tmp, dest)
}

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

// pushVersionToS3 uploads model + meta files for a specific version to
// S3 under the versions/{version}/ prefix so they survive restarts.
func pushVersionToS3(version string) {
	if S3Client == nil {
		return
	}
	modelsDir := resolveModelsDir()
	versionDir := filepath.Join(modelsDir, "versions", version)
	ctx := context.Background()

	for _, name := range append(modelFileNames, "meta.json") {
		localPath := filepath.Join(versionDir, name)
		f, err := os.Open(localPath)
		if err != nil {
			continue
		}
		stat, _ := f.Stat()
		s3Key := "versions/" + version + "/" + name
		_, err = S3Client.PutObject(ctx, ModelsBucket, s3Key, f, stat.Size(),
			minio.PutObjectOptions{ContentType: "application/octet-stream"})
		f.Close()
		if err != nil {
			log.Printf("[ModelS3] upload %s failed: %v", s3Key, err)
		} else {
			log.Printf("[ModelS3] uploaded %s (%d bytes)", s3Key, stat.Size())
		}
	}
}

// fetchFileFromS3 downloads an arbitrary S3 key to a local path.
func fetchFileFromS3(key, dest string) error {
	ctx := context.Background()
	obj, err := S3Client.GetObject(ctx, ModelsBucket, key, minio.GetObjectOptions{})
	if err != nil {
		return err
	}
	defer obj.Close()
	tmp := dest + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err = io.Copy(f, obj); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	f.Close()
	return os.Rename(tmp, dest)
}

// restoreVersionsFromS3 downloads meta.json for any versions that exist in S3
// but not locally — called at list time so version history survives restarts.
func restoreVersionsFromS3() {
	if S3Client == nil {
		return
	}
	ctx := context.Background()
	modelsDir := resolveModelsDir()
	versionsDir := filepath.Join(modelsDir, "versions")

	objects := S3Client.ListObjects(ctx, ModelsBucket, minio.ListObjectsOptions{
		Prefix:    "versions/",
		Recursive: true,
	})

	for obj := range objects {
		if obj.Err != nil {
			continue
		}
		rel := strings.TrimPrefix(obj.Key, "versions/")
		parts := strings.SplitN(rel, "/", 2)
		if len(parts) != 2 || parts[1] != "meta.json" {
			continue
		}
		version := parts[0]
		if !validVersionName(version) {
			continue
		}
		localMeta := filepath.Join(versionsDir, version, "meta.json")
		if fileExists(localMeta) {
			continue
		}
		if err := os.MkdirAll(filepath.Join(versionsDir, version), 0o755); err != nil {
			continue
		}
		if err := fetchFileFromS3(obj.Key, localMeta); err != nil {
			log.Printf("[ModelS3] restore version %s meta.json: %v", version, err)
		} else {
			log.Printf("[ModelS3] restored version %s from S3", version)
		}
	}
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

// validVersionName guards against path traversal in the version path segment.
func validVersionName(v string) bool {
	if v == "" {
		return false
	}
	for _, r := range v {
		if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' || r == '-') {
			return false
		}
	}
	return true
}

// PUT /api/models/upload/:version/:filename — AI worker uploads a freshly
// trained model file. Body is the raw file bytes. Saved to
// data/models/versions/{version}/{filename}. The worker calls this for each
// model file after training, before signalling "done" over gRPC.
func uploadModelFile(c *fiber.Ctx) error {
	version := c.Params("version")
	name := c.Params("filename")

	if !validVersionName(version) {
		return c.Status(400).JSON(fiber.Map{"error": "invalid version"})
	}

	allowed := false
	for _, n := range modelFileNames {
		if n == name {
			allowed = true
			break
		}
	}
	if !allowed {
		return c.Status(403).JSON(fiber.Map{"error": "file not allowed"})
	}

	body := c.Body()
	if len(body) == 0 {
		return c.Status(400).JSON(fiber.Map{"error": "empty body"})
	}

	destDir := filepath.Join(resolveModelsDir(), "versions", version)
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}
	dest := filepath.Join(destDir, name)
	if err := os.WriteFile(dest, body, 0o644); err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	log.Printf("[ModelUpload] saved %s for version %s (%d bytes)", name, version, len(body))
	return c.JSON(fiber.Map{"saved": name, "version": version, "bytes": len(body)})
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
