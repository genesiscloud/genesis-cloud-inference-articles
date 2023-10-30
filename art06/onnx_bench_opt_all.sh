#!/bin/bash

python3 onnx_bench.py -m ./onnx/bert_base_uncased_b1_s16_opt.onnx -b 1 -s 16
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b1_s64_opt.onnx -b 1 -s 64
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b1_s512_opt.onnx -b 1 -s 512

python3 onnx_bench.py -m ./onnx/bert_base_uncased_b8_s16_opt.onnx -b 8 -s 16
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b8_s64_opt.onnx -b 8 -s 64
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b8_s512_opt.onnx -b 8 -s 512

python3 onnx_bench.py -m ./onnx/bert_base_uncased_b64_s16_opt.onnx -b 64 -s 16
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b64_s64_opt.onnx -b 64 -s 64
python3 onnx_bench.py -m ./onnx/bert_base_uncased_b64_s512_opt.onnx -b 64 -s 512


