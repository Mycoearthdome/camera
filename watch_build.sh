#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:/home/jordan/NN_CAMERA/SERVER/TensorRT/TAR/TensorRT-10.15.1.29/lib

TRT=/home/jordan/NN_CAMERA/SERVER/TensorRT/TAR/TensorRT-10.15.1.29/bin/trtexec

while true; do
    if [ model.onnx -nt model.engine ]; then
        echo "Rebuilding engine..."
        $TRT --onnx=model.onnx --saveEngine=model.engine --fp16
    fi
    sleep 90
done
