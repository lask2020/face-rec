package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"

	"github.com/gofiber/fiber/v2"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

var (
	S3Client        *minio.Client
	FacesBucket     = "faces"
	SnapshotsBucket = "snapshots"
	ModelsBucket    = "models"
)

func InitS3() {
	endpoint := os.Getenv("S3_ENDPOINT")
	if endpoint == "" {
		endpoint = "rustfs:9000"
	}
	accessKey := os.Getenv("S3_ACCESS_KEY")
	if accessKey == "" {
		accessKey = "admin"
	}
	secretKey := os.Getenv("S3_SECRET_KEY")
	if secretKey == "" {
		secretKey = "admin12345"
	}

	faces := os.Getenv("S3_FACES_BUCKET")
	if faces != "" {
		FacesBucket = faces
	}
	snapshots := os.Getenv("S3_SNAPSHOTS_BUCKET")
	if snapshots != "" {
		SnapshotsBucket = snapshots
	}
	if mb := os.Getenv("S3_MODELS_BUCKET"); mb != "" {
		ModelsBucket = mb
	}

	log.Printf("Initializing S3 Client connected to %s...", endpoint)

	var err error
	S3Client, err = minio.New(endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(accessKey, secretKey, ""),
		Secure: false, // Local communication is over HTTP
	})
	if err != nil {
		log.Fatalf("Failed to initialize S3 client: %v", err)
	}

	// Ensure buckets exist
	ctx := context.Background()
	ensureBucketExists(ctx, FacesBucket)
	ensureBucketExists(ctx, SnapshotsBucket)
	ensureBucketExists(ctx, ModelsBucket)

	log.Println("S3 Client initialized successfully.")
}

func ensureBucketExists(ctx context.Context, bucketName string) {
	exists, err := S3Client.BucketExists(ctx, bucketName)
	if err != nil {
		log.Printf("[S3] Error checking bucket %s: %v", bucketName, err)
		return
	}
	if !exists {
		err = S3Client.MakeBucket(ctx, bucketName, minio.MakeBucketOptions{})
		if err != nil {
			log.Fatalf("[S3] Failed to create bucket %s: %v", bucketName, err)
		}
		log.Printf("[S3] Created bucket %s", bucketName)
	} else {
		log.Printf("[S3] Bucket %s already exists", bucketName)
	}
}

// GetSnapshotFromS3 streams the snapshot image from RustFS to the client
func GetSnapshotFromS3(c *fiber.Ctx) error {
	filename := c.Params("filename")
	if filename == "" {
		return c.Status(400).JSON(fiber.Map{"error": "Filename is required"})
	}

	object, err := S3Client.GetObject(c.Context(), SnapshotsBucket, filename, minio.GetObjectOptions{})
	if err != nil {
		return c.Status(404).SendString("File not found")
	}
	defer object.Close()

	// Check if file exists by reading its stat
	info, err := object.Stat()
	if err != nil {
		return c.Status(404).SendString("File not found")
	}

	c.Set("Content-Type", info.ContentType)
	c.Set("Content-Length", strconv.FormatInt(info.Size, 10))

	// Stream object using io.Copy
	_, err = io.Copy(c.Response().BodyWriter(), object)
	return err
}

// GetFaceFromS3 streams the face image from RustFS to the client
func GetFaceFromS3(c *fiber.Ctx) error {
	filename := c.Params("filename")
	if filename == "" {
		return c.Status(400).JSON(fiber.Map{"error": "Filename is required"})
	}

	object, err := S3Client.GetObject(c.Context(), FacesBucket, filename, minio.GetObjectOptions{})
	if err != nil {
		return c.Status(404).SendString("File not found")
	}
	defer object.Close()

	info, err := object.Stat()
	if err != nil {
		// Fallback to local file if not found in S3 (for transition period)
		localPath := fmt.Sprintf("./data/faces/%s", filename)
		if _, statErr := os.Stat(localPath); statErr == nil {
			return c.SendFile(localPath)
		}
		return c.Status(404).SendString("File not found")
	}

	c.Set("Content-Type", info.ContentType)
	c.Set("Content-Length", strconv.FormatInt(info.Size, 10))

	_, err = io.Copy(c.Response().BodyWriter(), object)
	return err
}
