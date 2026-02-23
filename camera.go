package main

import (
	"encoding/binary"
	"image"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"gocv.io/x/gocv"
)

const (
	Width     = 640
	Height    = 480
	FrameSize = Width * Height * 3
	ChunkSize = 1200
)

func main() {
	serverAddr := "192.168.1.151:5000" // change to your server IP
	conn, err := net.Dial("udp", serverAddr)
	if err != nil {
		log.Fatal(err)
	}
	defer conn.Close()

	webcam, err := gocv.OpenVideoCapture(0)
	if err != nil {
		log.Fatal(err)
	}
	defer webcam.Close()

	webcam.Set(gocv.VideoCaptureFrameWidth, Width)
	webcam.Set(gocv.VideoCaptureFrameHeight, Height)

	img := gocv.NewMat()
	defer img.Close()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	var frameID uint32 = 0
	log.Println("Capturing 24-bit BGR video...")

loop:
	for {
		select {
		case <-stop:
			break loop
		default:
			if ok := webcam.Read(&img); !ok || img.Empty() {
				continue
			}

			// resize to exact dimensions
			gocv.Resize(img, &img, image.Point{X: Width, Y: Height}, 0, 0, gocv.InterpolationDefault)
			quality := []int{gocv.IMWriteJpegQuality, 80} // 80% quality is a good balance
			buf, err := gocv.IMEncodeWithParams(gocv.JPEGFileExt, img, quality)
			if err != nil {
				log.Printf("Encode error: %v\n", err)
				continue
			}
			payload := buf.GetBytes()
			totalSize := uint32(len(payload))

			for offset := 0; offset < len(payload); offset += ChunkSize {
				end := offset + ChunkSize
				if end > len(payload) {
					end = len(payload)
				}
				chunkSize := end - offset

				// New header: 12 bytes
				headerBuf := make([]byte, 12+chunkSize)
				binary.BigEndian.PutUint32(headerBuf[0:4], frameID)
				binary.BigEndian.PutUint32(headerBuf[4:8], uint32(offset))
				binary.BigEndian.PutUint32(headerBuf[8:12], totalSize) // Send total JPEG size
				copy(headerBuf[12:], payload[offset:end])

				conn.Write(headerBuf)
				if err != nil {
					log.Printf("UDP send error: %v\n", err)
				}
			}

			frameID++
			time.Sleep(5 * time.Millisecond) // give NIC/CPU a breather
		}
	}

	log.Println("Client exiting...")
}
