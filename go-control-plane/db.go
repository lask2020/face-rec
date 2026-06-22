package main

import (
	"fmt"
	"log"
	"os"
	"time"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
)

var DB *gorm.DB

type Person struct {
	ID         uint         `gorm:"primaryKey" json:"id"`
	Name       string       `gorm:"size:255;not null;index" json:"name"`
	Department string       `gorm:"size:255;default:''" json:"department"`
	Notes      string       `gorm:"type:text;default:''" json:"notes"`
	CreatedAt  time.Time    `json:"created_at"`
	UpdatedAt  time.Time    `json:"updated_at"`
	Faces      []PersonFace `json:"faces" gorm:"foreignKey:PersonID"`
}

type PersonFace struct {
	ID        uint      `gorm:"primaryKey" json:"id"`
	PersonID  uint      `gorm:"not null;index" json:"person_id"`
	ImagePath string    `gorm:"size:512;not null" json:"image_path"`
	ImageUrl  string    `gorm:"-" json:"image_url"`
	Embedding []byte    `gorm:"type:bytea;not null" json:"-"`
	CreatedAt time.Time `json:"created_at"`
}

func (pf *PersonFace) AfterFind(tx *gorm.DB) (err error) {
	pf.ImageUrl = "/api/static/faces/" + pf.ImagePath
	return nil
}

func (pf *PersonFace) AfterSave(tx *gorm.DB) (err error) {
	pf.ImageUrl = "/api/static/faces/" + pf.ImagePath
	return nil
}

type Camera struct {
	ID         uint      `gorm:"primaryKey" json:"id"`
	Name       string    `gorm:"size:255;not null" json:"name"`
	URL        string    `gorm:"column:url;size:1024;not null" json:"url"`
	Location   string    `gorm:"size:255;default:''" json:"location"`
	IsActive   bool      `gorm:"default:false" json:"is_active"`
	FPSProcess int       `gorm:"default:2" json:"fps_process"`
	DetectMode string    `gorm:"size:16;default:'face'" json:"detect_mode"` // "face" | "plate" | "both"
	CreatedAt  time.Time `json:"created_at"`
}

func (Camera) TableName() string {
	return "cameras"
}

type DetectionLog struct {
	ID               uint      `gorm:"primaryKey" json:"id"`
	PersonID         *uint     `gorm:"index" json:"person_id"`
	PersonName       string    `gorm:"size:255;default:'Unknown'" json:"person_name"`
	CameraID         uint      `gorm:"not null;index" json:"camera_id"`
	CameraName       string    `gorm:"size:255;default:''" json:"camera_name"`
	Confidence       float64   `gorm:"default:0.0" json:"confidence"`
	SnapshotPath     string    `gorm:"size:512" json:"snapshot_url"`
	FaceCropPath     string    `gorm:"size:512" json:"face_crop_url"`
	RestoredFacePath string    `gorm:"size:512" json:"restored_face_url"`
	DetectedAt       time.Time `gorm:"index" json:"detected_at"`
}

func (DetectionLog) TableName() string {
	return "detection_logs"
}

type PlateDetectionLog struct {
	ID           uint      `gorm:"primaryKey" json:"id"`
	CameraID     uint      `gorm:"not null;index" json:"camera_id"`
	CameraName   string    `gorm:"size:255;default:''" json:"camera_name"`
	PlateNumber  string    `gorm:"size:32;default:''" json:"plate_number"`
	RawText      string    `gorm:"size:64;default:''" json:"raw_text"`
	Confidence   float64   `gorm:"default:0.0" json:"confidence"`
	PlateType    string    `gorm:"size:32;default:'unknown'" json:"plate_type"`
	Province     string    `gorm:"size:64;default:''" json:"province"`
	SnapshotPath string    `gorm:"size:512" json:"snapshot_url"`
	DetectedAt   time.Time `gorm:"index" json:"detected_at"`
}

func (PlateDetectionLog) TableName() string {
	return "plate_detection_logs"
}

// PlateTrainingSample stores individual plate crops for active-learning review.
type PlateTrainingSample struct {
	ID            uint      `gorm:"primaryKey" json:"id"`
	CameraID      uint      `gorm:"not null;index" json:"camera_id"`
	CameraName    string    `gorm:"size:255;default:''" json:"camera_name"`
	TrackID       string    `gorm:"size:64;default:'';index" json:"track_id"` // groups frames from the same PlateTrack
	ImagePath     string    `gorm:"size:512" json:"-"`          // S3 key — never mutated after save
	ImageURL      string    `gorm:"-" json:"image_url"`         // computed by AfterFind
	CharLabels    string    `gorm:"type:text;default:'[]'" json:"char_labels"` // JSON array
	RawText       string    `gorm:"size:64;default:''" json:"raw_text"`
	CorrectedText string    `gorm:"size:64;default:''" json:"corrected_text"`
	Confidence    float64   `gorm:"default:0.0" json:"confidence"`
	Status        string    `gorm:"size:16;default:'pending';index" json:"status"` // pending|approved|rejected
	DetectedAt    time.Time `gorm:"index" json:"detected_at"`
	CreatedAt     time.Time `json:"created_at"`
}

func (PlateTrainingSample) TableName() string {
	return "plate_training_samples"
}

func (pts *PlateTrainingSample) AfterFind(tx *gorm.DB) (err error) {
	if pts.ImagePath != "" {
		pts.ImageURL = "/api/static/snapshots/" + pts.ImagePath
	}
	return nil
}

func InitDatabase() {
	user := os.Getenv("POSTGRES_USER")
	pass := os.Getenv("POSTGRES_PASSWORD")
	dbName := os.Getenv("POSTGRES_DB")
	host := os.Getenv("POSTGRES_HOST")
	port := os.Getenv("POSTGRES_PORT")

	if host == "" {
		host = "localhost"
	}
	if port == "" {
		port = "5432"
	}
	if user == "" {
		user = "root"
	}
	if pass == "" {
		pass = "password"
	}
	if dbName == "" {
		dbName = "facerec"
	}

	dsn := fmt.Sprintf("host=%s user=%s password=%s dbname=%s port=%s sslmode=disable",
		host, user, pass, dbName, port)

	var err error
	DB, err = gorm.Open(postgres.Open(dsn), &gorm.Config{})
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}

	err = DB.AutoMigrate(&Person{}, &PersonFace{}, &Camera{}, &DetectionLog{}, &PlateDetectionLog{}, &PlateTrainingSample{})
	if err != nil {
		log.Fatalf("Failed to migrate database: %v", err)
	}

	log.Println("Database initialized and migrated.")
}
