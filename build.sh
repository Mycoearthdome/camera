#!/bin/bash
set -e

# Paths
CUDA_INC=/usr/local/cuda-12.8/targets/x86_64-linux/include
CUDA_LIB=/usr/local/cuda-12.8/targets/x86_64-linux/lib
TRT_INC=/home/jordan/NN_CAMERA/SERVER/TensorRT/TAR/TensorRT-10.15.1.29/include
TRT_LIB=/home/jordan/NN_CAMERA/SERVER/TensorRT/TAR/TensorRT-10.15.1.29/lib
EGL_INC=/usr/include/EGL
DRM_INC=/usr/include/libdrm

export LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:/home/jordan/NN_CAMERA/SERVER/TensorRT/TAR/TensorRT-10.15.1.29/lib:/home/jordan/NN_CAMERA/SERVER:/usr/include/libdrm

# Compile server_pipe.cu
/usr/local/cuda-12.8/bin/nvcc -c server_pipe.cu -arch=compute_86 -code=sm_86 \
    -o server_pipe.o \
    -Xcompiler -fPIC -lEGL -lGL -lX11 -lnvinfer -lcudart \
    -I$CUDA_INC -I$TRT_INC -I$EGL_INC -I$DRM_INC -I.

# Compile main.cpp
g++ -c main.cpp -o main.o -I$CUDA_INC -I$TRT_INC -I$EGL_INC -I$DRM_INC.

# Link everything
g++ main.o server_pipe.o -o server \
    -L$CUDA_LIB -L$TRT_LIB -L/usr/lib64 \
    -lcudart -lGL -lX11 -lEGL -lnvinfer -lcudart \
    -lnvonnxparser -lnvinfer_plugin \
    -lpthread -ldl -std=c++17

echo "Build complete: ./server"
