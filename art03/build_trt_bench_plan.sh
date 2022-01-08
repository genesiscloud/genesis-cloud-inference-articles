#!/bin/bash

mkdir -p ./bin

g++ -o ./bin/trt_bench_plan \
    -I /usr/local/cuda/include \
    trt_bench_plan.cpp common.cpp \
    -L /usr/local/cuda/lib64 -lnvinfer -lcudart


