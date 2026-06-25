package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/minio/minio-go/v7"
)

const geminiAPIURL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent"

// ── Gemini request/response types ─────────────────────────────────────────────

type geminiPart struct {
	Text       string           `json:"text,omitempty"`
	InlineData *geminiImageData `json:"inline_data,omitempty"`
}

type geminiImageData struct {
	MimeType string `json:"mime_type"`
	Data     string `json:"data"`
}

type geminiContent struct {
	Parts []geminiPart `json:"parts"`
}

type geminiRequest struct {
	Contents         []geminiContent  `json:"contents"`
	GenerationConfig *geminiGenConfig `json:"generationConfig,omitempty"`
}

type geminiGenConfig struct {
	ResponseMIMEType string `json:"responseMimeType"`
}

type geminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// ── Gemini helper ──────────────────────────────────────────────────────────────

func callGeminiVision(apiKey, rawText string, imageB64 string) (suggestion string, correctedText *string, reason string, err error) {
	prompt := fmt.Sprintf(`You are a Thai license plate OCR quality reviewer.

I will show you a cropped image of a Thai license plate.
The OCR system read this text from the plate: "%s"

Thai license plate rules:
- Format: 2 Thai consonants + 1-4 digits + optional province name
- Example: กข 1234 กรุงเทพ

Your task:
1. Look at the image carefully
2. Decide if the OCR text correctly represents the plate you see
3. If the plate is unreadable, too blurry, or clearly wrong → suggest "reject"
4. If the plate looks correct or nearly correct → suggest "approve"

Respond ONLY with valid JSON (no markdown, no code block), exactly:
{"suggestion":"approve","corrected_text":null,"reason":"brief reason"}

If you see a correction: {"suggestion":"approve","corrected_text":"กข-1234","reason":"brief reason"}`, rawText)

	reqBody := geminiRequest{
		Contents: []geminiContent{
			{
				Parts: []geminiPart{
					{
						InlineData: &geminiImageData{
							MimeType: "image/jpeg",
							Data:     imageB64,
						},
					},
					{Text: prompt},
				},
			},
		},
		GenerationConfig: &geminiGenConfig{
			ResponseMIMEType: "application/json",
		},
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return "", nil, "", err
	}

	url := geminiAPIURL + "?key=" + apiKey
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Post(url, "application/json", bytes.NewReader(bodyBytes))
	if err != nil {
		return "", nil, "", err
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", nil, "", err
	}

	var gemResp geminiResponse
	if err := json.Unmarshal(respBytes, &gemResp); err != nil {
		return "", nil, "", fmt.Errorf("parse response: %w", err)
	}

	if gemResp.Error != nil {
		return "", nil, "", fmt.Errorf("gemini API error: %s", gemResp.Error.Message)
	}

	if len(gemResp.Candidates) == 0 || len(gemResp.Candidates[0].Content.Parts) == 0 {
		return "", nil, "", fmt.Errorf("empty gemini response")
	}

	text := strings.TrimSpace(gemResp.Candidates[0].Content.Parts[0].Text)

	var parsed struct {
		Suggestion    string  `json:"suggestion"`
		CorrectedText *string `json:"corrected_text"`
		Reason        string  `json:"reason"`
	}
	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		return "", nil, "", fmt.Errorf("parse result JSON: %w, raw: %s", err, text)
	}

	return parsed.Suggestion, parsed.CorrectedText, parsed.Reason, nil
}

// ── REST Handler ───────────────────────────────────────────────────────────────

type AIReviewResult struct {
	ID            uint    `json:"id"`
	Suggestion    string  `json:"suggestion"` // "approve" | "reject" | "error"
	CorrectedText *string `json:"corrected_text"`
	Reason        string  `json:"reason"`
}

func aiReviewTrainingSamples(c *fiber.Ctx) error {
	var body struct {
		IDs []uint `json:"ids"`
	}
	if err := c.BodyParser(&body); err != nil || len(body.IDs) == 0 {
		return c.Status(400).JSON(fiber.Map{"error": "ids required"})
	}
	if len(body.IDs) > 50 {
		return c.Status(400).JSON(fiber.Map{"error": "max 50 samples per request"})
	}

	apiKey := getSettingValue("gemini_api_key")
	if apiKey == "" {
		return c.Status(400).JSON(fiber.Map{"error": "gemini_api_key not configured — save it in Settings"})
	}

	// Fetch samples from DB
	var samples []PlateTrainingSample
	if err := DB.Where("id IN ?", body.IDs).Find(&samples).Error; err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}

	ctx := context.Background()
	results := make([]AIReviewResult, 0, len(samples))

	for _, s := range samples {
		result := AIReviewResult{ID: s.ID}

		if s.ImagePath == "" {
			result.Suggestion = "error"
			result.Reason = "no image available"
			results = append(results, result)
			continue
		}

		// Fetch image bytes from S3
		obj, err := S3Client.GetObject(ctx, SnapshotsBucket, s.ImagePath, minio.GetObjectOptions{})
		if err != nil {
			result.Suggestion = "error"
			result.Reason = "image fetch failed: " + err.Error()
			results = append(results, result)
			continue
		}

		imgBytes, err := io.ReadAll(obj)
		obj.Close()
		if err != nil {
			result.Suggestion = "error"
			result.Reason = "image read failed: " + err.Error()
			results = append(results, result)
			continue
		}

		imageB64 := base64.StdEncoding.EncodeToString(imgBytes)

		suggestion, correctedText, reason, err := callGeminiVision(apiKey, s.RawText, imageB64)
		if err != nil {
			result.Suggestion = "error"
			result.Reason = err.Error()
		} else {
			result.Suggestion = suggestion
			result.CorrectedText = correctedText
			result.Reason = reason
		}

		results = append(results, result)
	}

	return c.JSON(fiber.Map{"results": results})
}
