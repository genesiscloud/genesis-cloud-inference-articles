#!/bin/bash

mkdir -p ./bin

g++ -o ./bin/trt_onnx_parser \
    -I /usr/local/cuda/include \
    trt_onnx_parser.cpp common.cpp \
    -L /usr/local/cuda/lib64 -lnvonnxparser -lnvinfer -lcudart


