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

// AICharLabel is the per-character bbox that Gemini returns.
// Matches the char_labels JSON schema stored in PlateTrainingSample.CharLabels.
type AICharLabel struct {
	ClassName  string  `json:"class_name"`
	CX         float64 `json:"cx"`
	CY         float64 `json:"cy"`
	BW         float64 `json:"bw"`
	BH         float64 `json:"bh"`
	Confidence float64 `json:"confidence"`
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
	correctedLabels []AICharLabel,
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

Coordinate system: YOLO-normalized (0.0–1.0) relative to the full image.
  cx, cy = center of box  |  bw, bh = width and height of box
  cx=0 is left edge, cx=1 is right edge, cy=0 is top, cy=1 is bottom

Valid class names:
- Digits: 0 1 2 3 4 5 6 7 8 9
- Thai consonants (exact Thai char): ก ข ค ฆ ง จ ฉ ช ซ ฌ ญ ฎ ฏ ฐ ฑ ฒ ณ ด ต ถ ท ธ น บ ป ผ ฝ พ ฟ ภ ม ย ร ล ว ศ ษ ส ห ฬ อ ฮ
- Province codes: ACR ATG AYA BKK BKN BRM CBI CCO CMI CNT CPM CPN CRI CTI KBI KKN KPT KRI KSN LEI LPG LPN LRI MDH MKM NAN NBI NBP NKI NMA NPM NPT NRT NSN NST NWT NYK PBI PCT PKN PKT PLG PLK PNA PNB PRE PRI PTN PTE PYO RBR RET RNG RYG SBR SKA SKM SKN SKW SNI SNK SPB SPK SRI SRN SSK STI STN TAK TRG TRT UBN UDN UTI UTT YLA YST

Your tasks:
1. Look at the plate image carefully
2. Return ALL character boxes (left→right order):
   - For EXISTING boxes: correct the class_name if wrong; adjust cx/cy/bw/bh if clearly misaligned
   - For MISSING characters (not detected by OCR): ADD a new box with accurate cx/cy/bw/bh
   - Set confidence=1.0 for boxes you are certain about, 0.7 for uncertain
3. Suggest "approve" if the plate is readable, "reject" if too blurry/unreadable

Respond ONLY with valid JSON (no markdown), exactly:
{"suggestion":"approve","corrected_text":"2กท-2459","corrected_labels":[{"class_name":"2","cx":0.07,"cy":0.50,"bw":0.11,"bh":0.75,"confidence":1.0},...],"reason":"brief reason"}

Rules:
- corrected_labels must be sorted left→right by cx
- If you cannot read the plate, set suggestion="reject" and corrected_labels=null
- corrected_text should be the assembled plate string you see in the image`, rawText, len(existing), charSummary)

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
		Suggestion      string        `json:"suggestion"`
		CorrectedText   *string       `json:"corrected_text"`
		CorrectedLabels []AICharLabel `json:"corrected_labels"`
		Reason          string        `json:"reason"`
	}
	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		return "", nil, nil, "", fmt.Errorf("parse result JSON: %w, raw: %s", err, text)
	}

	// Sanity-check coordinates — discard any label with out-of-range values.
	var valid []AICharLabel
	for _, l := range parsed.CorrectedLabels {
		if l.CX >= 0 && l.CX <= 1 && l.CY >= 0 && l.CY <= 1 &&
			l.BW > 0 && l.BW <= 1 && l.BH > 0 && l.BH <= 1 &&
			l.ClassName != "" {
			if l.Confidence == 0 {
				l.Confidence = 1.0
			}
			valid = append(valid, l)
		}
	}

	return parsed.Suggestion, parsed.CorrectedText, valid, parsed.Reason, nil
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
	ID              uint          `json:"id"`
	Suggestion      string        `json:"suggestion"` // "approve" | "reject" | "error"
	CorrectedText   *string       `json:"corrected_text"`
	CorrectedLabels []AICharLabel `json:"corrected_labels"` // nil if unavailable
	Reason          string        `json:"reason"`
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

		suggestion, correctedText, correctedLabels, reason, err := callGeminiVision(apiKey, s.RawText, charLabels, imageB64)
		if err != nil {
			result.Suggestion = "error"
			result.Reason = err.Error()
		} else {
			result.Suggestion = suggestion
			result.CorrectedText = correctedText
			result.CorrectedLabels = correctedLabels
			result.Reason = reason
		}

		results = append(results, result)
	}

	return c.JSON(fiber.Map{"results": results})
}
