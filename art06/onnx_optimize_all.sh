#!/bin/bash

python3 ./onnx_optimize.py -m bert-base-uncased -i ./onnx/bert_base_uncased_b1_s16.onnx -o ./onnx/bert_base_uncased_b1_s16_opt.onnx
python3 ./onnx_optimize.py -m bert-base-uncased -i ./onnx/bert_base_uncased_b1_s64.onnx -o ./onnx/bert_base_uncased_b1_s64_opt.onnx
python3 ./onnx_optimize.py -m bert-base-uncased -i ./onnx/bert_base_uncased_b1_s512.onnx -o ./onnx/bert_base_uncased_b1_s512_opt.onnx

python3 ./onnx_optimize.py -m bert-base-uncased -i ./onnx/bert_base_uncased_b8_s16.onnx -o ./onnx/bert_base_uncased_b8_s16_opt.onnx
python3 ./onnx_optimize.py -m bert-base-uncased -i ./onnx/bert_base_uncased_b8_s64.onnx -o ./onnx/bert_base_uncased_b8_s64_opt.onnx
python3 ./onnx_optimize.py -m bert-base-uncased -i ./onnx/bert_base_uncased_b8_s512.onnx -o ./onnx/bert_base_uncased_b8_s512_opt.onnx

python3 ./onnx_optimize.py -m bert-base-uncased -i ./onnx/bert_base_uncased_b64_s16.onnx -o ./onnx/bert_base_uncased_b64_s16_opt.onnx
python3 ./onnx_optimize.py -m bert-base-uncased -i ./onnx/bert_base_uncased_b64_s64.onnx -o ./onnx/bert_base_uncased_b64_s64_opt.onnx
python3 ./onnx_optimize.py -m bert-base-uncased -i ./onnx/bert_base_uncased_b64_s512.onnx -o ./onnx/bert_base_uncased_b64_s512_opt.onnx

