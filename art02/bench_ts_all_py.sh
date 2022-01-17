#!/bin/bash

echo "#head;TorchScript (Python)"

python3 bench_model_ts.py alexnet
python3 bench_model_ts.py densenet121
python3 bench_model_ts.py densenet161
python3 bench_model_ts.py densenet169
python3 bench_model_ts.py densenet201
python3 bench_model_ts.py mnasnet0_5
python3 bench_model_ts.py mnasnet1_0
python3 bench_model_ts.py mobilenet_v2
python3 bench_model_ts.py mobilenet_v3_large
python3 bench_model_ts.py mobilenet_v3_small
python3 bench_model_ts.py resnet101
python3 bench_model_ts.py resnet152
python3 bench_model_ts.py resnet18
python3 bench_model_ts.py resnet34
python3 bench_model_ts.py resnet50
python3 bench_model_ts.py resnext101_32x8d
python3 bench_model_ts.py resnext50_32x4d
python3 bench_model_ts.py shufflenet_v2_x0_5
python3 bench_model_ts.py shufflenet_v2_x1_0
python3 bench_model_ts.py squeezenet1_0
python3 bench_model_ts.py squeezenet1_1
python3 bench_model_ts.py vgg11
python3 bench_model_ts.py vgg11_bn
python3 bench_model_ts.py vgg13
python3 bench_model_ts.py vgg13_bn
python3 bench_model_ts.py vgg16
python3 bench_model_ts.py vgg16_bn
python3 bench_model_ts.py vgg19
python3 bench_model_ts.py vgg19_bn
python3 bench_model_ts.py wide_resnet101_2
python3 bench_model_ts.py wide_resnet50_2


