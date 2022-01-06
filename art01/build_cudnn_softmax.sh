#!/bin/bash

mkdir -p ./bin

g++ -o ./bin/cudnn_softmax -I /usr/local/cuda/include \
    cudnn_softmax.cpp \
    -L /usr/local/cuda/lib64 -lcudnn -lcudart

