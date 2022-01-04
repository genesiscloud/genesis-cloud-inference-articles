#!/bin/bash

mkdir -p ./bin

g++ -o ./bin/bench_ts \
    -I ~/vendor/libtorch/include \
    -I ~/vendor/libtorch/include/torch/csrc/api/include \
    bench_ts.cpp \
    -L ~/vendor/libtorch/lib \
    -lc10_cuda -lc10 \
    -Wl,--no-as-needed -ltorch_cuda -Wl,--as-needed \
    -ltorch_cpu -ltorch

