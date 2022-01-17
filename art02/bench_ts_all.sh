#!/bin/bash

echo "#head;TorchScript (C++)"

./bin/bench_ts ts/alexnet.ts
./bin/bench_ts ts/densenet121.ts
./bin/bench_ts ts/densenet161.ts
./bin/bench_ts ts/densenet169.ts
./bin/bench_ts ts/densenet201.ts
./bin/bench_ts ts/mnasnet0_5.ts
./bin/bench_ts ts/mnasnet1_0.ts
./bin/bench_ts ts/mobilenet_v2.ts
./bin/bench_ts ts/mobilenet_v3_large.ts
./bin/bench_ts ts/mobilenet_v3_small.ts
./bin/bench_ts ts/resnet101.ts
./bin/bench_ts ts/resnet152.ts
./bin/bench_ts ts/resnet18.ts
./bin/bench_ts ts/resnet34.ts
./bin/bench_ts ts/resnet50.ts
./bin/bench_ts ts/resnext101_32x8d.ts
./bin/bench_ts ts/resnext50_32x4d.ts
./bin/bench_ts ts/shufflenet_v2_x0_5.ts
./bin/bench_ts ts/shufflenet_v2_x1_0.ts
./bin/bench_ts ts/squeezenet1_0.ts
./bin/bench_ts ts/squeezenet1_1.ts
./bin/bench_ts ts/vgg11.ts
./bin/bench_ts ts/vgg11_bn.ts
./bin/bench_ts ts/vgg13.ts
./bin/bench_ts ts/vgg13_bn.ts
./bin/bench_ts ts/vgg16.ts
./bin/bench_ts ts/vgg16_bn.ts
./bin/bench_ts ts/vgg19.ts
./bin/bench_ts ts/vgg19_bn.ts
./bin/bench_ts ts/wide_resnet101_2.ts
./bin/bench_ts ts/wide_resnet50_2.ts


