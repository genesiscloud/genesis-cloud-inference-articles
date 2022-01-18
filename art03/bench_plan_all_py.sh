#!/bin/bash

echo "#head;TensorRT (Python)"

python3 trt_bench_plan.py ./plan/alexnet.plan
python3 trt_bench_plan.py ./plan/densenet121.plan
python3 trt_bench_plan.py ./plan/densenet161.plan
python3 trt_bench_plan.py ./plan/densenet169.plan
python3 trt_bench_plan.py ./plan/densenet201.plan
python3 trt_bench_plan.py ./plan/mnasnet0_5.plan
python3 trt_bench_plan.py ./plan/mnasnet1_0.plan
python3 trt_bench_plan.py ./plan/mobilenet_v2.plan
python3 trt_bench_plan.py ./plan/mobilenet_v3_large.plan
python3 trt_bench_plan.py ./plan/mobilenet_v3_small.plan
python3 trt_bench_plan.py ./plan/resnet101.plan
python3 trt_bench_plan.py ./plan/resnet152.plan
python3 trt_bench_plan.py ./plan/resnet18.plan
python3 trt_bench_plan.py ./plan/resnet34.plan
python3 trt_bench_plan.py ./plan/resnet50.plan
python3 trt_bench_plan.py ./plan/resnext101_32x8d.plan
python3 trt_bench_plan.py ./plan/resnext50_32x4d.plan
python3 trt_bench_plan.py ./plan/shufflenet_v2_x0_5.plan
python3 trt_bench_plan.py ./plan/shufflenet_v2_x1_0.plan
python3 trt_bench_plan.py ./plan/squeezenet1_0.plan
python3 trt_bench_plan.py ./plan/squeezenet1_1.plan
python3 trt_bench_plan.py ./plan/vgg11.plan
python3 trt_bench_plan.py ./plan/vgg11_bn.plan
python3 trt_bench_plan.py ./plan/vgg13.plan
python3 trt_bench_plan.py ./plan/vgg13_bn.plan
python3 trt_bench_plan.py ./plan/vgg16.plan
python3 trt_bench_plan.py ./plan/vgg16_bn.plan
python3 trt_bench_plan.py ./plan/vgg19.plan
python3 trt_bench_plan.py ./plan/vgg19_bn.plan
python3 trt_bench_plan.py ./plan/wide_resnet101_2.plan
python3 trt_bench_plan.py ./plan/wide_resnet50_2.plan


