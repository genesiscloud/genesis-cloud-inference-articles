#!/bin/bash

mkdir -p ./bin

g++ -O3 -o ./bin/onnx_bench \
    -I ~/vendor/onnxruntime/include \
    onnx_bench.cpp \
    -L ~/vendor/onnxruntime/lib \
    -lonnxruntime

