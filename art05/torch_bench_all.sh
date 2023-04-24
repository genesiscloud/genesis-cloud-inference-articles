#!/bin/bash

python3 torch_bench.py -m bert-base-uncased -b 1 -s 16
python3 torch_bench.py -m bert-base-uncased -b 1 -s 64
python3 torch_bench.py -m bert-base-uncased -b 1 -s 512

python3 torch_bench.py -m bert-base-uncased -b 8 -s 16
python3 torch_bench.py -m bert-base-uncased -b 8 -s 64
python3 torch_bench.py -m bert-base-uncased -b 8 -s 512

python3 torch_bench.py -m bert-base-uncased -b 64 -s 16
python3 torch_bench.py -m bert-base-uncased -b 64 -s 64
python3 torch_bench.py -m bert-base-uncased -b 64 -s 512


