package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// LandmarkData stores the vector and identity of a file
type LandmarkData struct {
	Filename  string
	Landmarks []float64
}

// getJpegComment finds the 0xFFFE marker and extracts the string
func getJpegComment(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	data, err := io.ReadAll(reader)
	if err != nil {
		return "", err
	}

	// JPEG markers start with 0xFF. COM marker is 0xFFFE.
	comMarker := []byte{0xFF, 0xFE}
	idx := bytes.Index(data, comMarker)
	if idx == -1 {
		return "", fmt.Errorf("no COM marker found")
	}

	// The 2 bytes after the marker are the length of the comment (Big Endian)
	length := int(data[idx+2])<<8 | int(data[idx+3])
	// The actual string starts after the 2-byte length field
	comment := data[idx+4 : idx+2+length]

	return string(comment), nil
}

func parseLandmarks(raw string) []float64 {
	strVals := strings.Split(raw, ",")
	landmarks := make([]float64, 0, len(strVals))
	for _, s := range strVals {
		val, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
		if err == nil {
			landmarks = append(landmarks, val)
		}
	}
	return landmarks
}

func calculateSimilarity(l1, l2 []float64) float64 {
	// Only compare vectors of the same size (e.g., both 120 or both 150)
	if len(l1) == 0 || len(l1) != len(l2) {
		return 0.0
	}

	var sumSquares float64
	for i := range l1 {
		diff := l1[i] - l2[i]
		sumSquares += diff * diff
	}
	distance := math.Sqrt(sumSquares)

	// Normalized Euclidean Similarity
	maxDist := math.Sqrt(float64(len(l1)))
	return math.Max(0, (1.0-(distance/maxDist))*100)
}

func main() {
	searchDir := "../pictures" // The folder created by your C server
	var library []LandmarkData

	// 1. Walk the directory and build a library of landmarks
	err := filepath.Walk(searchDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.ToLower(filepath.Ext(path)) == ".jpg" {
			raw, err := getJpegComment(path)
			if err == nil {
				points := parseLandmarks(raw)
				if len(points) > 0 {
					library = append(library, LandmarkData{Filename: path, Landmarks: points})
				}
			}
		}
		return nil
	})

	if err != nil {
		fmt.Printf("Error walking path: %v\n", err)
		return
	}

	fmt.Printf("Loaded %d images with metadata. Correlating...\n\n", len(library))

	// 2. Perform N x N correlation

	for i := 0; i < len(library); i++ {
		for j := i + 1; j < len(library); j++ {
			sim := calculateSimilarity(library[i].Landmarks, library[j].Landmarks)

			// Only report strong correlations (e.g., > 85%)
			if sim > 85.0 {
				fmt.Printf("[MATCH %.2f%%]\n  A: %s\n  B: %s\n\n", sim, library[i].Filename, library[j].Filename)
			}
		}
	}
}
