#!/bin/bash

echo "#head;ORT (C++)"

./bin/onnx_bench 1 16
./bin/onnx_bench 1 64
./bin/onnx_bench 1 512
./bin/onnx_bench 8 16
./bin/onnx_bench 8 64
./bin/onnx_bench 8 512
./bin/onnx_bench 64 16
./bin/onnx_bench 64 64
./bin/onnx_bench 64 512

./bin/onnx_bench 1 16 opt
./bin/onnx_bench 1 64 opt
./bin/onnx_bench 1 512 opt
./bin/onnx_bench 8 16 opt
./bin/onnx_bench 8 64 opt
./bin/onnx_bench 8 512 opt
./bin/onnx_bench 64 16 opt
./bin/onnx_bench 64 64 opt
./bin/onnx_bench 64 512 opt

