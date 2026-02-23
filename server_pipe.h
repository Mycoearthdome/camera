// server_pipe.h
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

int server_init(const char* engine_path, int W, int H);
int server_process_frame(unsigned char* host_frame, int size);
void server_cleanup(void);

#ifdef __cplusplus
}
#endif