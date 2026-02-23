#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include "server_pipe.h"  // your CUDA/GL pipeline
#include <jpeglib.h>
#include <setjmp.h>

#define UDP_PORT    5000
#define CHUNK_SIZE  1200

volatile sig_atomic_t keep_running = 1;



typedef struct {
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
} jpeg_error_mgr_ext;

static void jpeg_error_exit(j_common_ptr cinfo) {
    jpeg_error_mgr_ext *err = (jpeg_error_mgr_ext *) cinfo->err;
    longjmp(err->setjmp_buffer, 1);
}

/*
 * Decode JPEG from memory into BGR format.
 *
 * Input:
 *   jpeg_data   - pointer to JPEG bytes
 *   jpeg_size   - size of JPEG buffer
 *
 * Output:
 *   out_width
 *   out_height
 *   out_channels (always 3)
 *
 * Return:
 *   Pointer to BGR buffer (malloc'd). Caller must free().
 *   NULL on failure.
 */
uint8_t* jpeg_to_bgr(
    const uint8_t *jpeg_data,
    size_t jpeg_size,
    int *out_width,
    int *out_height,
    int *out_channels)
{
    struct jpeg_decompress_struct cinfo;
    jpeg_error_mgr_ext jerr;

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpeg_error_exit;

    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        return NULL;
    }

    jpeg_create_decompress(&cinfo);

    // Read JPEG from memory
    #if JPEG_LIB_VERSION >= 80 || defined(MEM_SRCDST_SUPPORTED)
        jpeg_mem_src(&cinfo, jpeg_data, jpeg_size);
    #else
    #error "jpeg_mem_src not supported by this libjpeg version"
    #endif

    jpeg_read_header(&cinfo, TRUE);

    // Force RGB output
    cinfo.out_color_space = JCS_RGB;

    jpeg_start_decompress(&cinfo);

    int width = cinfo.output_width;
    int height = cinfo.output_height;
    int channels = cinfo.output_components; // should be 3

    size_t row_stride = width * channels;

    uint8_t *rgb_buffer = (uint8_t*)malloc(width * height * channels);
    if (!rgb_buffer) {
        jpeg_destroy_decompress(&cinfo);
        return NULL;
    }

    // Read scanlines
    while (cinfo.output_scanline < cinfo.output_height) {
        uint8_t *rowptr = rgb_buffer + cinfo.output_scanline * row_stride;
        jpeg_read_scanlines(&cinfo, &rowptr, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    // Convert RGB → BGR in-place
    for (int i = 0; i < width * height; i++) {
        uint8_t *p = rgb_buffer + i * 3;
        uint8_t tmp = p[0];
        p[0] = p[2];
        p[2] = tmp;
    }

    *out_width = width;
    *out_height = height;
    *out_channels = 3;

    return rgb_buffer;
}



void handle_sigint(int sig) { keep_running = 0; }



int main(int argc, char** argv) {
    const char* engine_path = "model.engine";
    int W = 640, H = 480;
    if (argc > 1) engine_path = argv[1];

    // ---------------- Initialize pipeline ----------------
    if (server_init(engine_path, W, H) != 0) {
        fprintf(stderr, "Failed to init server pipeline\n");
        return -1;
    }

    // ---------------- Setup UDP socket ----------------
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) { perror("socket"); server_cleanup(); return -1; }

    struct sockaddr_in addr, client;
    socklen_t client_len = sizeof(client);
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(UDP_PORT);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(sockfd); server_cleanup(); return -1;
    }

    // ---------------- SIGINT handler ----------------
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);

    // ---------------- Frame buffer ----------------
    size_t FRAME_SIZE = W * H * 3;
    unsigned char* frame_buffer = (unsigned char*)malloc(FRAME_SIZE);
    if (!frame_buffer) { perror("malloc"); close(sockfd); server_cleanup(); return -1; }

    size_t bytes_received = 0;
    uint32_t current_frame = 0;

    printf("Server listening on UDP port %d...\n", UDP_PORT);

    // ---------------- Main loop ----------------
    while (keep_running) {
        unsigned char buf[12 + CHUNK_SIZE]; 
        int n = recvfrom(sockfd, buf, sizeof(buf), 0, (struct sockaddr*)&client, &client_len);
        if (n < 12) continue; 

        uint32_t frame_id   = ntohl(*(uint32_t*)(buf + 0));
        uint32_t offset     = ntohl(*(uint32_t*)(buf + 4));
        uint32_t total_size = ntohl(*(uint32_t*)(buf + 8));
        uint32_t chunk_size = n - 12;

        // Reset buffer if a new frame starts
        if (frame_id != current_frame) {
            bytes_received = 0;
            current_frame = frame_id;
        }

        // Copy payload into assembly buffer
        if (offset + chunk_size <= FRAME_SIZE) {
            memcpy(frame_buffer + offset, buf + 12, chunk_size);
            bytes_received += chunk_size;
        }

        // Frame complete: Trigger JPEG decompression
        if (bytes_received >= total_size && total_size > 0) {
            int outW, outH, outC;
            
            // Use the function already defined in your main.c
            uint8_t* bgr_pixels = jpeg_to_bgr(frame_buffer, total_size, &outW, &outH, &outC);
            
            if (bgr_pixels) {
                server_process_frame(bgr_pixels, outW * outH * outC);
                free(bgr_pixels); // Important: jpeg_to_bgr uses malloc
            }
            
            bytes_received = 0;
            current_frame++; 
        }
    }
    printf("\nShutting down server...\n");
    close(sockfd);
    server_cleanup();
    free(frame_buffer);
    return 0;
}