package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

var RDB *redis.Client
var ctx = context.Background()

// --- Detection Deduplication ---
// Prevents logging the same person on the same camera within a cooldown window.
var (
	detectionCooldown = 60 * time.Second // Only log same person+camera once per 60 seconds
	lastDetections    = make(map[string]time.Time)
	lastDetMu         sync.Mutex
)

// dedupKey returns a unique key for person+camera combo
func dedupKey(cameraID uint, personID *uint) string {
	pid := "unknown"
	if personID != nil {
		pid = fmt.Sprintf("%d", *personID)
	}
	return fmt.Sprintf("%d:%s", cameraID, pid)
}

// shouldLog returns true if this detection should be recorded (not a duplicate)
func shouldLog(cameraID uint, personID *uint) bool {
	lastDetMu.Lock()
	defer lastDetMu.Unlock()

	key := dedupKey(cameraID, personID)
	if last, exists := lastDetections[key]; exists {
		if time.Since(last) < detectionCooldown {
			return false // Too soon, skip
		}
	}
	lastDetections[key] = time.Now()
	return true
}

func InitRedis() {
	addr := os.Getenv("REDIS_URL")
	if addr == "" {
		addr = "localhost:6379"
	}

	RDB = redis.NewClient(&redis.Options{
		Addr: addr,
	})

	_, err := RDB.Ping(ctx).Result()
	if err != nil {
		log.Fatalf("Failed to connect to Redis: %v", err)
	}

	log.Println("Redis initialized.")
}

func PublishAssignment(cameraID uint, rtspURL string, fps int) error {
	payload := fmt.Sprintf("start|%d|%s|%d", cameraID, rtspURL, fps)
	return RDB.Publish(ctx, "worker.assign", payload).Err()
}

func PublishStopCommand(cameraID uint) error {
	payload := fmt.Sprintf("stop|%d|", cameraID)
	return RDB.Publish(ctx, "worker.assign", payload).Err()
}

// RestartActiveCameras queries all cameras with is_active = true from DB and publishes start commands.
func RestartActiveCameras() {
	var activeCameras []Camera
	if err := DB.Where("is_active = ?", true).Find(&activeCameras).Error; err != nil {
		log.Printf("[Startup] Failed to query active cameras: %v", err)
		return
	}

	if len(activeCameras) == 0 {
		log.Println("[Startup] No active cameras to restart.")
		return
	}

	log.Printf("[Startup] Found %d active cameras to restart. Waiting for services to settle...", len(activeCameras))
	
	// Wait a small moment to ensure services (Redis, ingestion) are fully ready
	time.Sleep(3 * time.Second)

	for _, cam := range activeCameras {
		log.Printf("[Startup] Automatically restarting camera stream for '%s' (ID: %d, FPS: %d)...", cam.Name, cam.ID, cam.FPSProcess)
		if err := PublishAssignment(cam.ID, cam.URL, cam.FPSProcess); err != nil {
			log.Printf("[Startup] Failed to publish assignment for camera %d: %v", cam.ID, err)
		}
	}
}

