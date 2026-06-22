package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"mime"
	"mime/multipart"
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

// captureLoop consumes a continuous MJPEG stream from go2rtc and forwards frames
// to Redis. It maintains ONE long-lived connection per camera instead of polling
// frame.jpeg repeatedly.
//
// Why a stream instead of snapshot polling:
//   - frame.jpeg is an on-demand decode; polling it at high fps (e.g. 12) makes
//     go2rtc decode-per-request, fall behind, serve stale duplicate frames, and
//     eventually freeze.
//   - stream.mjpeg lets go2rtc decode once at source rate and push frames to us.
//     We read fresh frames continuously and throttle pushes to the desired fps.
func captureLoop(ctx context.Context, camID uint, streamName string, intervalMs int) {
	log.Printf("[Camera %d] Starting MJPEG capture (target %dms interval) from go2rtc stream '%s'", camID, intervalMs, streamName)

	mjpegURL := fmt.Sprintf("%s/api/stream.mjpeg?src=%s", go2rtcURL, streamName)
	minInterval := time.Duration(intervalMs) * time.Millisecond

	for {
		select {
		case <-ctx.Done():
			log.Printf("[Camera %d] Capture stopped.", camID)
			return
		default:
		}

		err := streamMJPEG(ctx, camID, mjpegURL, minInterval)
		if ctx.Err() != nil {
			log.Printf("[Camera %d] Capture stopped.", camID)
			return
		}
		log.Printf("[Camera %d] MJPEG stream ended (%v). Reconnecting in 2s...", camID, err)
		select {
		case <-ctx.Done():
			return
		case <-time.After(2 * time.Second):
		}
	}
}

// streamMJPEG opens a single MJPEG (multipart/x-mixed-replace) connection and
// reads JPEG frames until the stream errors or stalls. It throttles forwarding
// to minInterval (the configured fps) while keeping the connection drained so
// go2rtc never backs up. A watchdog reconnects if go2rtc freezes mid-stream.
func streamMJPEG(parentCtx context.Context, camID uint, url string, minInterval time.Duration) error {
	ctx, cancel := context.WithCancel(parentCtx)
	defer cancel()

	req, _ := http.NewRequestWithContext(ctx, "GET", url, nil)
	resp, err := http.DefaultClient.Do(req) // no client timeout — this is a long-lived stream
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("go2rtc returned %d", resp.StatusCode)
	}

	mediaType, params, err := mime.ParseMediaType(resp.Header.Get("Content-Type"))
	if err != nil || !strings.HasPrefix(mediaType, "multipart/") {
		return fmt.Errorf("not a multipart stream: %q", resp.Header.Get("Content-Type"))
	}
	mr := multipart.NewReader(resp.Body, params["boundary"])

	// Watchdog: if no frame arrives within stallTimeout, cancel the request so the
	// blocked read unblocks and the caller reconnects. Recovers from go2rtc freezes.
	const stallTimeout = 5 * time.Second
	watchdog := time.AfterFunc(stallTimeout, cancel)
	defer watchdog.Stop()

	var lastPush time.Time
	for {
		part, err := mr.NextPart()
		if err != nil {
			return err
		}
		frame, err := io.ReadAll(part)
		part.Close()
		if err != nil {
			return err
		}
		watchdog.Reset(stallTimeout)

		if len(frame) < 100 {
			continue
		}

		// Throttle forwarding to the configured fps. We still read every part above
		// so the TCP stream stays drained and go2rtc keeps delivering fresh frames.
		now := time.Now()
		if now.Sub(lastPush) < minInterval {
			continue
		}
		lastPush = now

		// Live preview for the UI
		rdb.Set(ctx, fmt.Sprintf("camera:latest:%d", camID), frame, 10*time.Second)
		pushFrameToAIQueue(ctx, camID, frame)
	}
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
