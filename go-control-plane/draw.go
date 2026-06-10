package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
)

// DrawBBoxesOnJPEG draws rectangles on a JPEG image.
// bboxes: slice of [x1, y1, x2, y2]
// isKnown: slice of bool representing if the detection is associated with a registered person.
func DrawBBoxesOnJPEG(jpegBytes []byte, bboxes [][]float64, isKnown []bool, quality int) ([]byte, error) {
	if len(bboxes) == 0 {
		return jpegBytes, nil
	}

	srcImg, err := jpeg.Decode(bytes.NewReader(jpegBytes))
	if err != nil {
		return nil, err
	}

	bounds := srcImg.Bounds()
	rgbaImg := image.NewRGBA(bounds)
	draw.Draw(rgbaImg, bounds, srcImg, bounds.Min, draw.Src)

	for i, bbox := range bboxes {
		if len(bbox) < 4 {
			continue
		}
		x1, y1, x2, y2 := int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

		// Keep coordinates within image bounds
		x1 = max(0, min(x1, bounds.Max.X-1))
		x2 = max(0, min(x2, bounds.Max.X-1))
		y1 = max(0, min(y1, bounds.Max.Y-1))
		y2 = max(0, min(y2, bounds.Max.Y-1))

		// Color: Green for known, Orange for unknown
		col := color.RGBA{0, 255, 0, 255} // Green
		if !isKnown[i] {
			col = color.RGBA{255, 165, 0, 255} // Orange
		}

		// Draw rectangle with thickness 3
		drawBBox(rgbaImg, x1, y1, x2, y2, col, 3)
	}

	var buf bytes.Buffer
	err = jpeg.Encode(&buf, rgbaImg, &jpeg.Options{Quality: quality})
	if err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

func drawBBox(img *image.RGBA, x1, y1, x2, y2 int, col color.Color, thickness int) {
	for t := 0; t < thickness; t++ {
		// Top and Bottom lines
		for x := x1 - t; x <= x2 + t; x++ {
			img.Set(x, y1-t, col)
			img.Set(x, y2+t, col)
		}
		// Left and Right lines
		for y := y1 - t; y <= y2 + t; y++ {
			img.Set(x1-t, y, col)
			img.Set(x2+t, y, col)
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// CropJPEG crops a portion of a JPEG image and returns the cropped JPEG bytes.
func CropJPEG(jpegBytes []byte, x1, y1, x2, y2 int, quality int) ([]byte, error) {
	srcImg, err := jpeg.Decode(bytes.NewReader(jpegBytes))
	if err != nil {
		return nil, err
	}

	bounds := srcImg.Bounds()
	x1 = max(0, min(x1, bounds.Max.X-1))
	x2 = max(0, min(x2, bounds.Max.X-1))
	y1 = max(0, min(y1, bounds.Max.Y-1))
	y2 = max(0, min(y2, bounds.Max.Y-1))

	if x1 >= x2 || y1 >= y2 {
		return nil, fmt.Errorf("invalid crop dimensions")
	}

	rect := image.Rect(x1, y1, x2, y2)

	// If the image supports SubImage (which YCbCr and RGBA do)
	if subImg, ok := srcImg.(interface {
		SubImage(r image.Rectangle) image.Image
	}); ok {
		croppedImg := subImg.SubImage(rect)
		var buf bytes.Buffer
		err = jpeg.Encode(&buf, croppedImg, &jpeg.Options{Quality: quality})
		if err != nil {
			return nil, err
		}
		return buf.Bytes(), nil
	}

	// Fallback: copy pixels manually
	rgbaImg := image.NewRGBA(image.Rect(0, 0, rect.Dx(), rect.Dy()))
	draw.Draw(rgbaImg, rgbaImg.Bounds(), srcImg, rect.Min, draw.Src)
	var buf bytes.Buffer
	err = jpeg.Encode(&buf, rgbaImg, &jpeg.Options{Quality: quality})
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

