#!/bin/bash

echo "#head;PyTorch"

python3 bench_model.py alexnet
python3 bench_model.py densenet121
python3 bench_model.py densenet161
python3 bench_model.py densenet169
python3 bench_model.py densenet201
python3 bench_model.py mnasnet0_5
python3 bench_model.py mnasnet1_0
python3 bench_model.py mobilenet_v2
python3 bench_model.py mobilenet_v3_large
python3 bench_model.py mobilenet_v3_small
python3 bench_model.py resnet101
python3 bench_model.py resnet152
python3 bench_model.py resnet18
python3 bench_model.py resnet34
python3 bench_model.py resnet50
python3 bench_model.py resnext101_32x8d
python3 bench_model.py resnext50_32x4d
python3 bench_model.py shufflenet_v2_x0_5
python3 bench_model.py shufflenet_v2_x1_0
python3 bench_model.py squeezenet1_0
python3 bench_model.py squeezenet1_1
python3 bench_model.py vgg11
python3 bench_model.py vgg11_bn
python3 bench_model.py vgg13
python3 bench_model.py vgg13_bn
python3 bench_model.py vgg16
python3 bench_model.py vgg16_bn
python3 bench_model.py vgg19
python3 bench_model.py vgg19_bn
python3 bench_model.py wide_resnet101_2
python3 bench_model.py wide_resnet50_2


