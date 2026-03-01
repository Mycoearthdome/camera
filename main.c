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
#include <time.h>
#include <sys/stat.h>
#include <errno.h>

#define UDP_PORT    5000
#define CHUNK_SIZE  1100

volatile sig_atomic_t keep_running = 1;

typedef struct {
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
} jpeg_error_mgr_ext;

// Extended error manager to prevent libjpeg from killing the server process on error
struct my_error_mgr {
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

static void jpeg_error_exit(j_common_ptr cinfo) {
    jpeg_error_mgr_ext *err = (jpeg_error_mgr_ext *) cinfo->err;
    longjmp(err->setjmp_buffer, 1);
}

// Helper to swap 64-bit doubles from Big Endian to Host Endian
double ntohd(const uint8_t* buf) {
    uint64_t temp;
    memcpy(&temp, buf, 8);
    temp = __builtin_bswap64(temp);
    double result;
    memcpy(&result, &temp, 8);
    return result;
}



static void my_error_exit(j_common_ptr cinfo) {
    struct my_error_mgr *myerr = (struct my_error_mgr *)cinfo->err;
    (*cinfo->err->output_message)(cinfo);
    longjmp(myerr->setjmp_buffer, 1);
}

void save_mosaic_with_metadata(uint8_t* bgr_pixels, int w, int h, double* landmarks, int landmark_count) {
    if (!bgr_pixels || w <= 0 || h <= 0) return;

    struct jpeg_compress_struct cinfo;
    struct my_error_mgr jerr;
    FILE *outfile = NULL;
    char *meta_str = NULL;
    char path[256];
    const char *folder = "pictures";

    // 1. Create directory if it doesn't exist
    // 0777 sets read/write/execute permissions for all users (modified by umask)
    if (mkdir(folder, 0777) == -1) {
        if (errno != EEXIST) {
            fprintf(stderr, "Error creating directory %s\n", folder);
            return;
        }
    }

    // 2. Precise Path Generation
    // Including the folder name in the snprintf path
    snprintf(path, sizeof(path), "%s/mosaic_%ld_%04x.jpg", folder, (long)time(NULL), rand() & 0xFFFF);

    // 3. Setup libjpeg Error Handling
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;
    if (setjmp(jerr.setjmp_buffer)) {
        if (outfile) fclose(outfile);
        if (meta_str) free(meta_str);
        jpeg_destroy_compress(&cinfo);
        fprintf(stderr, "Critical: libjpeg compression error for %s\n", path);
        return;
    }

    // 4. Robust String Serialization
    if (landmark_count > 0 && landmarks != NULL) {
        int needed = 0;
        for (int i = 0; i < landmark_count; i++) {
            needed += snprintf(NULL, 0, "%.4f%s", landmarks[i], (i == landmark_count - 1) ? "" : ",");
        }

        if (needed < 65530) {
            meta_str = (char*)malloc(needed + 1);
            if (meta_str) {
                int pos = 0;
                for (int i = 0; i < landmark_count; i++) {
                    pos += sprintf(meta_str + pos, "%.4f%s", landmarks[i], (i == landmark_count - 1) ? "" : ",");
                }
            }
        }
    }

    // 5. Initialization & Compression
    jpeg_create_compress(&cinfo);
    if ((outfile = fopen(path, "wb")) == NULL) {
        fprintf(stderr, "Error opening output file %s\n", path);
        if (meta_str) free(meta_str);
        jpeg_destroy_compress(&cinfo);
        return;
    }

    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = w;
    cinfo.image_height = h;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_EXT_BGR; 

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 90, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    // 6. Inject Metadata
    if (meta_str) {
        jpeg_write_marker(&cinfo, JPEG_COM, (const JOCTET*)meta_str, strlen(meta_str));
    }

    // 7. Memory-Efficient Row Writing
    int row_stride = w * 3;
    while (cinfo.next_scanline < cinfo.image_height) {
        JSAMPROW row_pointer = &bgr_pixels[cinfo.next_scanline * row_stride];
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }

    // 8. Cleanup
    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);
    if (meta_str) free(meta_str);

    printf("SUCCESS: Saved to %s [%d landmarks]\n", path, landmark_count);
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
    int W = 640, H = 640;
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
        // 1. Max packet size: Header(32) + MaxLandmarks(16*15*8) + Chunk(1100)
        // 3000 bytes is a safe buffer for jumbo/normal frames
        unsigned char buf[4096]; 
        int n = recvfrom(sockfd, buf, sizeof(buf), 0, (struct sockaddr*)&client, &client_len);
        
        if (n < 32) continue; // Minimum header size now 32

        // --- Parse Fixed Header ---
        uint32_t frame_id       = ntohl(*(uint32_t*)(buf + 0));
        uint32_t jpeg_offset    = ntohl(*(uint32_t*)(buf + 4));
        uint32_t jpeg_total_sz  = ntohl(*(uint32_t*)(buf + 8));
        double latitude         = ntohd(buf + 12);
        double longitude        = ntohd(buf + 20);
        uint32_t landmark_count = ntohl(*(uint32_t*)(buf + 28));

        // --- Calculate Dynamic Offsets ---
        uint32_t landmark_bytes = landmark_count * 8;
        uint32_t header_total   = 32 + landmark_bytes;
        uint32_t chunk_size     = n - header_total;

        // --- Reassemble JPEG ---
        if (frame_id > current_frame) {
            bytes_received = 0;
            current_frame = frame_id;
        }

        // Ensure we don't overflow the reassembly buffer
        if (jpeg_offset + chunk_size <= FRAME_SIZE) {
            // Copy JPEG data starting after the landmarks
            memcpy(frame_buffer + jpeg_offset, buf + header_total, chunk_size);
            bytes_received += chunk_size;
        }

        // --- Process Completed Frame ---
        if (jpeg_offset + chunk_size == jpeg_total_sz && bytes_received >= jpeg_total_sz) {
            int outW, outH, outC;
            uint8_t* bgr_pixels = jpeg_to_bgr(frame_buffer, jpeg_total_sz, &outW, &outH, &outC);
            
            if (bgr_pixels) {
                // 1. Extract landmarks from the current packet's buffer
                // (Note: In a production environment, you'd store these during reassembly 
                // if landmarks differ across packets, but usually they are redundant per frame)
                double* landmarks_to_save = (double*)malloc(landmark_count * sizeof(double));
                for(uint32_t l = 0; l < landmark_count; l++) {
                    landmarks_to_save[l] = ntohd(buf + 32 + (l * 8));
                }

                // 2. Save image and metadata
                save_mosaic_with_metadata(bgr_pixels, outW, outH, landmarks_to_save, landmark_count);
                
                // 3. Process for CUDA/GL pipeline
                server_process_frame(bgr_pixels, outW * outH * outC);
                
                free(landmarks_to_save);
                free(bgr_pixels); 
            }
            bytes_received = 0;
        }
    }
    printf("\nShutting down server...\n");
    close(sockfd);
    server_cleanup();
    free(frame_buffer);
    return 0;
}