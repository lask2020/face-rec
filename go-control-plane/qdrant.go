package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"time"
)

var (
	QdrantURL      = "http://qdrant:6333"
	CollectionName = "face_embeddings"
	httpClient     = &http.Client{Timeout: 10 * time.Second}
)

func InitQdrant() {
	url := os.Getenv("QDRANT_URL")
	if url != "" {
		QdrantURL = url
	}
	log.Printf("Qdrant REST URL set to: %s", QdrantURL)

	// Ensure the collection exists
	ctx := context.Background()
	err := ensureCollectionExists(ctx)
	if err != nil {
		log.Printf("Warning: Failed to verify Qdrant collection: %v", err)
	}
}

type QdrantErrorResponse struct {
	Status struct {
		Error string `json:"error"`
	} `json:"status"`
}

func ensureCollectionExists(ctx context.Context) error {
	checkURL := fmt.Sprintf("%s/collections/%s", QdrantURL, CollectionName)
	req, _ := http.NewRequestWithContext(ctx, "GET", checkURL, nil)
	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		log.Printf("[Qdrant] Collection '%s' exists", CollectionName)
		return nil
	}

	log.Printf("[Qdrant] Collection '%s' not found. Creating it...", CollectionName)
	createURL := fmt.Sprintf("%s/collections/%s", QdrantURL, CollectionName)
	createBody := map[string]interface{}{
		"vectors": map[string]interface{}{
			"size":     512,
			"distance": "Cosine",
		},
	}
	bodyBytes, _ := json.Marshal(createBody)

	req, _ = http.NewRequestWithContext(ctx, "PUT", createURL, bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	resp, err = httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("create collection failed with status %d: %s", resp.StatusCode, string(respBytes))
	}

	log.Printf("[Qdrant] Collection '%s' created successfully", CollectionName)
	return nil
}

type SearchResult struct {
	Id      uint    `json:"id"`
	Score   float64 `json:"score"`
	Payload struct {
		PersonID uint `json:"person_id"`
		FaceID   uint `json:"face_id"`
	} `json:"payload"`
}

type SearchResponse struct {
	Result []SearchResult `json:"result"`
}

// SearchFaceEmbedding searches Qdrant for the most similar face vector
func SearchFaceEmbedding(ctx context.Context, embedding []float32, threshold float64) (*uint, *uint, float64, error) {
	searchURL := fmt.Sprintf("%s/collections/%s/points/search", QdrantURL, CollectionName)
	searchBody := map[string]interface{}{
		"vector":          embedding,
		"limit":           1,
		"score_threshold": threshold,
		"with_payload":    true,
	}
	bodyBytes, _ := json.Marshal(searchBody)

	req, _ := http.NewRequestWithContext(ctx, "POST", searchURL, bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, nil, 0.0, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBytes, _ := io.ReadAll(resp.Body)
		return nil, nil, 0.0, fmt.Errorf("qdrant search failed: %s", string(respBytes))
	}

	var searchResp SearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&searchResp); err != nil {
		return nil, nil, 0.0, err
	}

	if len(searchResp.Result) == 0 {
		return nil, nil, 0.0, nil
	}

	hit := searchResp.Result[0]
	return &hit.Payload.PersonID, &hit.Payload.FaceID, hit.Score, nil
}

// AddFaceEmbedding adds or updates a face embedding in Qdrant
func AddFaceEmbedding(ctx context.Context, personID uint, faceID uint, embedding []float32) error {
	upsertURL := fmt.Sprintf("%s/collections/%s/points?wait=true", QdrantURL, CollectionName)
	upsertBody := map[string]interface{}{
		"points": []map[string]interface{}{
			{
				"id":     faceID,
				"vector": embedding,
				"payload": map[string]interface{}{
					"person_id": personID,
					"face_id":   faceID,
				},
			},
		},
	}
	bodyBytes, _ := json.Marshal(upsertBody)

	req, _ := http.NewRequestWithContext(ctx, "PUT", upsertURL, bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("qdrant upsert failed: %s", string(respBytes))
	}

	log.Printf("[Qdrant] Successfully added face ID %d for person ID %d", faceID, personID)
	return nil
}

// DeleteFaceEmbedding removes a single face embedding from Qdrant
func DeleteFaceEmbedding(ctx context.Context, faceID uint) error {
	deleteURL := fmt.Sprintf("%s/collections/%s/points/delete?wait=true", QdrantURL, CollectionName)
	deleteBody := map[string]interface{}{
		"points": []uint{faceID},
	}
	bodyBytes, _ := json.Marshal(deleteBody)

	req, _ := http.NewRequestWithContext(ctx, "POST", deleteURL, bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("qdrant delete face failed: %s", string(respBytes))
	}

	log.Printf("[Qdrant] Successfully deleted face ID %d", faceID)
	return nil
}

// DeletePersonEmbeddings removes all face embeddings belonging to a person
func DeletePersonEmbeddings(ctx context.Context, personID uint) error {
	deleteURL := fmt.Sprintf("%s/collections/%s/points/delete?wait=true", QdrantURL, CollectionName)
	deleteBody := map[string]interface{}{
		"filter": map[string]interface{}{
			"must": []map[string]interface{}{
				{
					"key": "person_id",
					"match": map[string]interface{}{
						"value": personID,
					},
				},
			},
		},
	}
	bodyBytes, _ := json.Marshal(deleteBody)

	req, _ := http.NewRequestWithContext(ctx, "POST", deleteURL, bytes.NewReader(bodyBytes))
	req.Header.Set("Content-Type", "application/json")
	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("qdrant delete person failed: %s", string(respBytes))
	}

	log.Printf("[Qdrant] Successfully deleted all face embeddings for person ID %d", personID)
	return nil
}
