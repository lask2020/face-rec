package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

var (
	rdb           *redis.Client
	ctx           = context.Background()
	activeCameras = make(map[uint]context.CancelFunc)
	activeMutex   sync.Mutex
	go2rtcURL     string
)

func main() {
	log.Println("Ingestion Worker started. Waiting for instructions...")

	redisAddr := os.Getenv("REDIS_URL")
	if redisAddr == "" {
		redisAddr = "localhost:6379"
	}
	go2rtcURL = os.Getenv("GO2RTC_URL")
	if go2rtcURL == "" {
		go2rtcURL = "http://localhost:1984"
	}

	rdb = redis.NewClient(&redis.Options{Addr: redisAddr})

	// Listen for camera assignments from Control Plane
	pubsub := rdb.Subscribe(ctx, "worker.assign")
	defer pubsub.Close()

	ch := pubsub.Channel()
	for msg := range ch {
		parts := strings.SplitN(msg.Payload, "|", 4)
		if len(parts) < 3 {
			log.Printf("Invalid message: %s", msg.Payload)
			continue
		}

		action := parts[0]
		camID64, _ := strconv.ParseUint(parts[1], 10, 32)
		camID := uint(camID64)
		rtspURL := parts[2]

		// Parse FPS from 4th field (default: 2 for backward compat)
		fps := 2
		if len(parts) >= 4 {
			if parsed, err := strconv.Atoi(parts[3]); err == nil && parsed > 0 {
				fps = parsed
			}
		}

		switch action {
		case "start":
			handleStart(camID, rtspURL, fps)
		case "stop":
			handleStop(camID)
		default:
			log.Printf("Unknown action: %s", action)
		}
	}
}

func handleStart(camID uint, rtspURL string, fps int) {
	activeMutex.Lock()
	defer activeMutex.Unlock()

	// Stop existing stream if running
	if cancel, exists := activeCameras[camID]; exists {
		cancel()
		delete(activeCameras, camID)
		log.Printf("[Camera %d] Restarting...", camID)
	}

	ictx, cancel := context.WithCancel(ctx)
	activeCameras[camID] = cancel

	// Register stream in go2rtc first (using direct RTSP URL without ffmpeg wrapper)
	streamName := fmt.Sprintf("cam_%d", camID)
	streamSource := rtspURL
	if err := registerStream(streamName, streamSource); err != nil {
		log.Printf("[Camera %d] Failed to register in go2rtc: %v. Will retry on capture.", camID, err)
	} else {
		log.Printf("[Camera %d] Registered in go2rtc as '%s' (Keyframe only)", camID, streamName)
	}

	// Convert FPS to millisecond interval (e.g. 3 fps -> 333ms)
	intervalMs := 1000 / fps
	go captureLoop(ictx, camID, streamName, intervalMs)
}

func handleStop(camID uint) {
	activeMutex.Lock()
	defer activeMutex.Unlock()

	if cancel, exists := activeCameras[camID]; exists {
		cancel()
		delete(activeCameras, camID)
		log.Printf("[Camera %d] Stream stopped.", camID)
	}
}

// registerStream tells go2rtc to connect to an RTSP/RTMP source.
// go2rtc will maintain the connection and share it with all consumers.
func registerStream(name, sourceURL string) error {
	apiURL := fmt.Sprintf("%s/api/streams?src=%s&name=%s",
		go2rtcURL,
		url.QueryEscape(sourceURL), // Encode & and other special chars in RTSP URL
		url.QueryEscape(name),
	)
	req, _ := http.NewRequest("PUT", apiURL, nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("go2rtc API error: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("go2rtc returned %d: %s", resp.StatusCode, string(body))
	}
	return nil
}

// captureLoop periodically grabs a snapshot from go2rtc and pushes it to Redis.
//
// This is extremely CPU-efficient because:
// 1. go2rtc handles the RTSP connection (persistent, no reconnection overhead)
// 2. go2rtc decodes exactly 1 frame when snapshot is requested
// 3. The ingestion worker just does an HTTP GET — no ffmpeg subprocess needed
func captureLoop(ctx context.Context, camID uint, streamName string, intervalMs int) {
	log.Printf("[Camera %d] Starting snapshot capture every %dms from go2rtc stream '%s'", camID, intervalMs, streamName)

	// Keep the go2rtc RTSP connection warm with a persistent consumer.
	// This is what makes frame.jpeg return instantly from go2rtc's buffer — without
	// it, every snapshot forces go2rtc to cold-start the RTSP pull, which is slow.
	// The consumer MUST auto-reconnect: a single network blip would otherwise kill
	// it permanently and every subsequent snapshot would lag.
	go keepStreamWarm(ctx, camID, streamName)

	// Use a Ticker so the interval is measured from the START of each cycle,
	// not from after the fetch+push complete. This keeps FPS accurate regardless
	// of how long the HTTP snapshot request takes.
	ticker := time.NewTicker(time.Duration(intervalMs) * time.Millisecond)
	defer ticker.Stop()

	// Snapshot timeout is kept tight (relative to the capture interval) so a single
	// stalled fetch can't freeze this camera's loop for many seconds.
	client := &http.Client{Timeout: 5 * time.Second}
	snapshotURL := fmt.Sprintf("%s/api/frame.jpeg?src=%s", go2rtcURL, streamName)

	for {
		select {
		case <-ctx.Done():
			log.Printf("[Camera %d] Capture stopped.", camID)
			return
		case <-ticker.C:
		}

		frame, err := fetchSnapshot(client, snapshotURL)
		if err != nil {
			log.Printf("[Camera %d] Snapshot failed: %v", camID, err)
			continue
		}

		// Always update live-preview for the UI
		rdb.Set(ctx, fmt.Sprintf("camera:latest:%d", camID), frame, 10*time.Second)

		pushFrameToAIQueue(ctx, camID, frame)
	}
}

// keepStreamWarm holds a long-lived consumer on the go2rtc stream so the RTSP
// source stays connected and decoded frames are always ready for snapshotting.
// It reconnects automatically until the camera's context is cancelled.
func keepStreamWarm(ctx context.Context, camID uint, streamName string) {
	url := fmt.Sprintf("%s/api/stream.mp4?src=%s", go2rtcURL, streamName)
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			// Stream not ready yet (or go2rtc restarting) — back off and retry.
			select {
			case <-ctx.Done():
				return
			case <-time.After(2 * time.Second):
			}
			continue
		}
		// Drain until the connection drops, then loop to reconnect.
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()

		select {
		case <-ctx.Done():
			return
		case <-time.After(1 * time.Second):
		}
	}
}

// fetchSnapshot grabs a single JPEG frame from go2rtc's snapshot API
func fetchSnapshot(client *http.Client, url string) ([]byte, error) {
	resp, err := client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("HTTP error: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("go2rtc returned %d", resp.StatusCode)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read error: %w", err)
	}

	if len(data) < 100 {
		return nil, fmt.Errorf("frame too small (%d bytes)", len(data))
	}

	return data, nil
}

// pushFrameToAIQueue pushes a JPEG frame to the AI processing stream.
// MaxLen with Approx bounds the queue and naturally drops the OLDEST entries when
// full — exactly the right behavior for real-time: fresh frames always win, stale
// ones are discarded. No producer-side length check is needed (and a producer-side
// skip would wrongly drop the NEW frame while keeping stale ones).
func pushFrameToAIQueue(ctx context.Context, camID uint, frame []byte) {
	err := rdb.XAdd(ctx, &redis.XAddArgs{
		Stream: "image.queue",
		MaxLen: 10, // small bound — keep the queue fresh, drop oldest under load
		Approx: true,
		Values: map[string]interface{}{
			"camera_id": camID,
			"data":      frame,
			"ts":        time.Now().UnixMilli(),
		},
	}).Err()

	if err != nil {
		log.Printf("[Camera %d] Failed to push to Redis Stream: %v", camID, err)
		return
	}

	log.Printf("[Camera %d] Frame captured (%d bytes), pushed to Redis.", camID, len(frame))
}
