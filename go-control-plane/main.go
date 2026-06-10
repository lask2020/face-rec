package main

import (
	"log"

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

	// Detections
	api.Get("/detections", listDetections)
	api.Get("/detections/stats", getDetectionStats)
	api.Get("/detections/overview", getOverviewStats)

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
	if err := app.Listen(":8000"); err != nil {
		log.Fatalf("Error starting server: %v", err)
	}
}
