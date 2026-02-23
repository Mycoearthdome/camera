#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include "server_pipe.h"  // your CUDA/GL pipeline

#define UDP_PORT    5000
#define CHUNK_SIZE  1200

volatile sig_atomic_t keep_running = 1;
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
        unsigned char buf[10 + CHUNK_SIZE];  // max packet size
        int n = recvfrom(sockfd, buf, sizeof(buf), 0,
                         (struct sockaddr*)&client, &client_len);
        if (n < 10) continue; // must have header

        // ---------------- Decode header ----------------
        uint32_t frame_id = ntohl(*(uint32_t*)(buf + 0));
        uint32_t offset   = ntohl(*(uint32_t*)(buf + 4));
        uint16_t size     = ntohs(*(uint16_t*)(buf + 8));

        // New frame?
        if (frame_id != current_frame) {
            bytes_received = 0;
            current_frame = frame_id;
        }

        // Bounds check
        if (size == 0 || size > CHUNK_SIZE || offset + size > FRAME_SIZE) {
            fprintf(stderr, "Packet invalid (offset=%u size=%u). Dropping.\n", offset, size);
            continue;
        }

        // Copy payload
        memcpy(frame_buffer + offset, buf + 10, size);
        if (offset + size > bytes_received)
            bytes_received = offset + size;

        // Frame complete
        if (bytes_received == FRAME_SIZE) {
            if (server_process_frame(frame_buffer, FRAME_SIZE) != 0)
            //if (server_process_frame(NULL, FRAME_SIZE) != 0) //DEBUG
                fprintf(stderr, "Failed to process frame\n");

            bytes_received = 0;
            current_frame++; // next frame
        }
    }

    printf("\nShutting down server...\n");
    close(sockfd);
    server_cleanup();
    free(frame_buffer);
    return 0;
}