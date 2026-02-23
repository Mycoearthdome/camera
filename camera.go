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
			payload := img.ToBytes() // guaranteed to be Width*Height*3

			// send in chunks
			for offset := 0; offset < len(payload); offset += ChunkSize {
				end := offset + ChunkSize
				if end > len(payload) {
					end = len(payload)
				}
				size := end - offset

				buf := make([]byte, 10+size)
				// header: frameID (4B) | offset (4B) | size (2B)
				binary.BigEndian.PutUint32(buf[0:4], frameID)
				binary.BigEndian.PutUint32(buf[4:8], uint32(offset))
				binary.BigEndian.PutUint16(buf[8:10], uint16(size))
				copy(buf[10:], payload[offset:end])

				_, err := conn.Write(buf)
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
