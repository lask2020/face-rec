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

// AICharLabel is a NEW bounding box that Gemini detected (missing from OCR).
// Matches the char_labels JSON schema stored in PlateTrainingSample.CharLabels.
type AICharLabel struct {
	ClassName  string  `json:"class_name"`
	CX         float64 `json:"cx"`
	CY         float64 `json:"cy"`
	BW         float64 `json:"bw"`
	BH         float64 `json:"bh"`
	Confidence float64 `json:"confidence"`
}

// AIReviewLabel is what callGeminiVision returns — corrected class names for
// existing boxes (keep positions) + new boxes that were missed entirely.
type AIReviewLabel struct {
	CorrectedClasses []string     // same length as input char_labels; nil = unchanged
	NewBoxes         []AICharLabel // genuinely missing characters with AI-estimated coords
}

// charLabelInput is used only for building the prompt.
type charLabelInput struct {
	ClassName  string  `json:"class_name"`
	CX         float64 `json:"cx"`
	CY         float64 `json:"cy"`
	BW         float64 `json:"bw"`
	BH         float64 `json:"bh"`
	Confidence float64 `json:"confidence"`
}

// ── Gemini helper ──────────────────────────────────────────────────────────────

func callGeminiVision(apiKey, rawText, charLabelsJSON string, imageB64 string) (
	suggestion string,
	correctedText *string,
	labels *AIReviewLabel,
	reason string,
	err error,
) {
	existing := parseCharLabels(charLabelsJSON)
	charSummary := buildCharSummary(existing)

	prompt := fmt.Sprintf(`You are a Thai license plate OCR quality reviewer.

I will show you a cropped image of a Thai license plate.
The OCR system detected: "%s"

Current character bounding boxes (%d boxes), sorted left→right:
%s
IMPORTANT: The bounding box POSITIONS above are accurate (from the real detector).
Do NOT change any cx/cy/bw/bh values for these existing boxes.

Valid class names:
- Digits: 0 1 2 3 4 5 6 7 8 9
- Thai consonants (exact Thai char): ก ข ค ฆ ง จ ฉ ช ซ ฌ ญ ฎ ฏ ฐ ฑ ฒ ณ ด ต ถ ท ธ น บ ป ผ ฝ พ ฟ ภ ม ย ร ล ว ศ ษ ส ห ฬ อ ฮ
- Province codes: ACR ATG AYA BKK BKN BRM CBI CCO CMI CNT CPM CPN CRI CTI KBI KKN KPT KRI KSN LEI LPG LPN LRI MDH MKM NAN NBI NBP NKI NMA NPM NPT NRT NSN NST NWT NYK PBI PCT PKN PKT PLG PLK PNA PNB PRE PRI PTN PTE PYO RBR RET RNG RYG SBR SKA SKM SKN SKW SNI SNK SPB SPK SRI SRN SSK STI STN TAK TRG TRT UBN UDN UTI UTT YLA YST

Coordinate system for new_boxes only: YOLO-normalized (0.0–1.0).
  cx, cy = center  |  bw, bh = size  |  cx=0 left, cx=1 right

Your tasks:
1. Look at the plate image carefully
2. For each existing box (left→right), correct the class_name if wrong → put in "corrected_classes" (must have exactly %d items)
3. If there are characters in the image NOT covered by any existing box → add them in "new_boxes" with estimated coordinates
4. Suggest "approve" if readable, "reject" if too blurry/unreadable

Respond ONLY with valid JSON (no markdown):
{"suggestion":"approve","corrected_text":"2กท-2459","corrected_classes":["2","ก","ท","2","4","5","9"],"new_boxes":[],"reason":"brief reason"}

Rules:
- corrected_classes MUST have exactly %d items (one per existing box, same order)
- new_boxes is [] when no characters are missing
- If unreadable: suggestion="reject", corrected_classes=null, new_boxes=null`, rawText, len(existing), charSummary, len(existing), len(existing))

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
		return "", nil, nil, "", err
	}

	url := geminiAPIURL + "?key=" + apiKey
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Post(url, "application/json", bytes.NewReader(bodyBytes))
	if err != nil {
		return "", nil, nil, "", err
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", nil, nil, "", err
	}

	var gemResp geminiResponse
	if err := json.Unmarshal(respBytes, &gemResp); err != nil {
		return "", nil, nil, "", fmt.Errorf("parse response: %w", err)
	}

	if gemResp.Error != nil {
		return "", nil, nil, "", fmt.Errorf("gemini API error: %s", gemResp.Error.Message)
	}

	if len(gemResp.Candidates) == 0 || len(gemResp.Candidates[0].Content.Parts) == 0 {
		return "", nil, nil, "", fmt.Errorf("empty gemini response")
	}

	text := strings.TrimSpace(gemResp.Candidates[0].Content.Parts[0].Text)

	var parsed struct {
		Suggestion       string        `json:"suggestion"`
		CorrectedText    *string       `json:"corrected_text"`
		CorrectedClasses []string      `json:"corrected_classes"`
		NewBoxes         []AICharLabel `json:"new_boxes"`
		Reason           string        `json:"reason"`
	}
	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		return "", nil, nil, "", fmt.Errorf("parse result JSON: %w, raw: %s", err, text)
	}

	// Only accept corrected_classes if count matches existing boxes.
	var result *AIReviewLabel
	if len(parsed.CorrectedClasses) == len(existing) || len(parsed.NewBoxes) > 0 {
		result = &AIReviewLabel{}
		if len(parsed.CorrectedClasses) == len(existing) {
			result.CorrectedClasses = parsed.CorrectedClasses
		}
		// Sanity-check new box coordinates.
		for _, b := range parsed.NewBoxes {
			if b.CX >= 0 && b.CX <= 1 && b.CY >= 0 && b.CY <= 1 &&
				b.BW > 0 && b.BW <= 1 && b.BH > 0 && b.BH <= 1 && b.ClassName != "" {
				if b.Confidence == 0 {
					b.Confidence = 0.8
				}
				result.NewBoxes = append(result.NewBoxes, b)
			}
		}
	}

	return parsed.Suggestion, parsed.CorrectedText, result, parsed.Reason, nil
}

// ── helpers ────────────────────────────────────────────────────────────────────

func parseCharLabels(charLabelsJSON string) []charLabelInput {
	var labels []charLabelInput
	_ = json.Unmarshal([]byte(charLabelsJSON), &labels)
	return labels
}

func buildCharSummary(labels []charLabelInput) string {
	if len(labels) == 0 {
		return "  (no detections)"
	}
	var sb strings.Builder
	for i, l := range labels {
		fmt.Fprintf(&sb, "  char %d: class=%q  cx=%.3f cy=%.3f bw=%.3f bh=%.3f  conf=%.0f%%\n",
			i+1, l.ClassName, l.CX, l.CY, l.BW, l.BH, l.Confidence*100)
	}
	return sb.String()
}

// ── REST Handler ───────────────────────────────────────────────────────────────

type AIReviewResult struct {
	ID               uint          `json:"id"`
	Suggestion       string        `json:"suggestion"`        // "approve" | "reject" | "error"
	CorrectedText    *string       `json:"corrected_text"`
	CorrectedClasses []string      `json:"corrected_classes"` // replaces class_name of existing boxes (positions unchanged)
	NewBoxes         []AICharLabel `json:"new_boxes"`         // newly detected missing chars with AI-estimated coords
	Reason           string        `json:"reason"`
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

		charLabels := s.CharLabels
		if charLabels == "" {
			charLabels = "[]"
		}

		suggestion, correctedText, reviewLabels, reason, err := callGeminiVision(apiKey, s.RawText, charLabels, imageB64)
		if err != nil {
			result.Suggestion = "error"
			result.Reason = err.Error()
		} else {
			result.Suggestion = suggestion
			result.CorrectedText = correctedText
			result.Reason = reason
			if reviewLabels != nil {
				result.CorrectedClasses = reviewLabels.CorrectedClasses
				result.NewBoxes = reviewLabels.NewBoxes
			}
		}

		results = append(results, result)
	}

	return c.JSON(fiber.Map{"results": results})
}
