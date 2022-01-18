#!/bin/bash

echo "#head;TensorRT (C++)"

./bin/trt_bench_plan ./plan/alexnet.plan
./bin/trt_bench_plan ./plan/densenet121.plan
./bin/trt_bench_plan ./plan/densenet161.plan
./bin/trt_bench_plan ./plan/densenet169.plan
./bin/trt_bench_plan ./plan/densenet201.plan
./bin/trt_bench_plan ./plan/mnasnet0_5.plan
./bin/trt_bench_plan ./plan/mnasnet1_0.plan
./bin/trt_bench_plan ./plan/mobilenet_v2.plan
./bin/trt_bench_plan ./plan/mobilenet_v3_large.plan
./bin/trt_bench_plan ./plan/mobilenet_v3_small.plan
./bin/trt_bench_plan ./plan/resnet101.plan
./bin/trt_bench_plan ./plan/resnet152.plan
./bin/trt_bench_plan ./plan/resnet18.plan
./bin/trt_bench_plan ./plan/resnet34.plan
./bin/trt_bench_plan ./plan/resnet50.plan
./bin/trt_bench_plan ./plan/resnext101_32x8d.plan
./bin/trt_bench_plan ./plan/resnext50_32x4d.plan
./bin/trt_bench_plan ./plan/shufflenet_v2_x0_5.plan
./bin/trt_bench_plan ./plan/shufflenet_v2_x1_0.plan
./bin/trt_bench_plan ./plan/squeezenet1_0.plan
./bin/trt_bench_plan ./plan/squeezenet1_1.plan
./bin/trt_bench_plan ./plan/vgg11.plan
./bin/trt_bench_plan ./plan/vgg11_bn.plan
./bin/trt_bench_plan ./plan/vgg13.plan
./bin/trt_bench_plan ./plan/vgg13_bn.plan
./bin/trt_bench_plan ./plan/vgg16.plan
./bin/trt_bench_plan ./plan/vgg16_bn.plan
./bin/trt_bench_plan ./plan/vgg19.plan
./bin/trt_bench_plan ./plan/vgg19_bn.plan
./bin/trt_bench_plan ./plan/wide_resnet101_2.plan
./bin/trt_bench_plan ./plan/wide_resnet50_2.plan


