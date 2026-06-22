package main

import (
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/websocket/v2"
)

func main() {
	app := fiber.New(fiber.Config{
		AppName: "Face Recognition Control Plane",
		// Increase body limit for face image uploads
		BodyLimit: 10 * 1024 * 1024,
	})

	app.Use(logger.New())
	app.Use(cors.New())

	InitDatabase()
	InitRedis()
	InitS3()
	InitQdrant()
	StartGRPCServer()

	// Automatically restart active cameras on boot
	go RestartActiveCameras()

	api := app.Group("/api")

	// Cameras
	api.Get("/cameras", GetCameras)
	api.Post("/cameras", AddCamera)
	api.Put("/cameras/:id", UpdateCamera)
	api.Delete("/cameras/:id", DeleteCamera)
	api.Post("/cameras/:id/start", StartCameraStream)
	api.Post("/cameras/:id/stop", StopCameraStream)
	api.Get("/cameras/:id/snapshot", GetCameraSnapshot)

	// Persons & Faces
	api.Get("/persons", GetPersons)
	api.Post("/persons", CreatePerson)
	api.Get("/persons/:id", GetPerson)
	api.Put("/persons/:id", UpdatePerson)
	api.Delete("/persons/:id", DeletePerson)
	api.Post("/persons/:id/faces", UploadFace)
	api.Delete("/persons/:id/faces/:face_id", DeleteFace)

	// Face Detections
	api.Get("/detections", listDetections)
	api.Get("/detections/stats", getDetectionStats)
	api.Get("/detections/overview", getOverviewStats)

	// License Plate Detections
	api.Get("/plate-detections", listPlateDetections)
	api.Get("/plate-detections/stats", getPlateDetectionStats)
	api.Delete("/plate-detections", clearAllPlateDetections)

	// Training Review
	api.Get("/training/plates", listTrainingSamples)
	api.Get("/training/plates/stats", getTrainingStats)
	api.Get("/training/plates/export", exportTrainingZip)
	api.Get("/training/plates/export/preview", getExportPreview)
	api.Get("/training/plates/:id", getTrainingSample)
	api.Put("/training/plates/:id", updateTrainingSample)
	api.Post("/training/plates/bulk", bulkUpdateTrainingSamples)

	// Workers
	api.Get("/workers", GetWorkers)
	api.Post("/workers/:id/toggle-pause", ToggleWorkerPauseHandler)

	// Surveillance Station
	api.Post("/surveillance-station/test", TestSSConnection)
	api.Post("/surveillance-station/cameras", ListSSCameras)
	api.Post("/surveillance-station/import", ImportSSCameras)

	// Static Assets (Proxied from S3)
	api.Get("/static/snapshots/:filename", GetSnapshotFromS3)
	api.Get("/static/faces/:filename", GetFaceFromS3)

	// WebSockets
	app.Get("/ws/events", websocket.New(EventsWebSocket))

	log.Println("Go Control Plane starting on port 8000...")

	// Start Fiber REST server in background
	go func() {
		if err := app.Listen(":8000"); err != nil {
			log.Printf("Fiber server listener error: %v", err)
		}
	}()

	// Block main thread waiting for exit signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	sig := <-sigChan
	log.Printf("Received signal: %v. Initiating graceful shutdown...", sig)

	// 1. Shutdown Fiber App
	log.Println("Shutting down Fiber REST server...")
	if err := app.Shutdown(); err != nil {
		log.Printf("Fiber shutdown error: %v", err)
	}

	// 2. Shutdown gRPC Server
	log.Println("Shutting down gRPC Server...")
	if grpcServer != nil {
		grpcServer.GracefulStop()
	}

	// 3. Close Redis Connection
	log.Println("Closing Redis connection...")
	if RDB != nil {
		RDB.Close()
	}

	// 4. Close DB Connection
	log.Println("Closing Database connection...")
	if DB != nil {
		sqlDB, err := DB.DB()
		if err == nil {
			sqlDB.Close()
		}
	}

	log.Println("Graceful shutdown completed successfully.")
}
