#!/bin/bash

mkdir -p ./bin

g++ -o ./bin/onnx_hello_world \
    -I ~/vendor/onnxruntime/include \
    onnx_hello_world.cpp \
    -L ~/vendor/onnxruntime/lib \
    -lonnxruntime

