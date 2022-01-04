#!/bin/bash

mkdir -p ./bin

g++ -o ./bin/infer_model_ts \
    -I ~/vendor/libtorch/include \
    -I ~/vendor/libtorch/include/torch/csrc/api/include \
    infer_model_ts.cpp \
    -L ~/vendor/libtorch/lib \
    -lc10_cuda -lc10 \
    -Wl,--no-as-needed -ltorch_cuda -Wl,--as-needed \
    -ltorch_cpu -ltorch


