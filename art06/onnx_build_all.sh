#!/bin/bash

python3 onnx_build.py -m bert-base-uncased -o ./onnx/bert_base_uncased_b1_s16.onnx -b 1 -s 16
python3 onnx_build.py -m bert-base-uncased -o ./onnx/bert_base_uncased_b1_s64.onnx -b 1 -s 64
python3 onnx_build.py -m bert-base-uncased -o ./onnx/bert_base_uncased_b1_s512.onnx -b 1 -s 512

python3 onnx_build.py -m bert-base-uncased -o ./onnx/bert_base_uncased_b8_s16.onnx -b 8 -s 16
python3 onnx_build.py -m bert-base-uncased -o ./onnx/bert_base_uncased_b8_s64.onnx -b 8 -s 64
python3 onnx_build.py -m bert-base-uncased -o ./onnx/bert_base_uncased_b8_s512.onnx -b 8 -s 512

python3 onnx_build.py -m bert-base-uncased -o ./onnx/bert_base_uncased_b64_s16.onnx -b 64 -s 16
python3 onnx_build.py -m bert-base-uncased -o ./onnx/bert_base_uncased_b64_s64.onnx -b 64 -s 64
python3 onnx_build.py -m bert-base-uncased -o ./onnx/bert_base_uncased_b64_s512.onnx -b 64 -s 512


