#!/bin/bash

python3 ./build.py \
    --no-container-build \
    --cmake-dir=/home/ubuntu/factory/server-2.18.0/build \
    --build-dir=/home/ubuntu/factory/server-2.18.0/scratch \
    --install-dir=/home/ubuntu/triton/server \
    --enable-logging \
    --enable-stats \
    --enable-tracing \
    --enable-metrics \
    --enable-gpu \
    --endpoint=http \
    --endpoint=grpc \
    --repo-tag=common:r22.01 \
    --repo-tag=core:r22.01 \
    --repo-tag=backend:r22.01 \
    --repo-tag=thirdparty:r22.01 \
    --backend=ensemble \
    --backend=tensorrt:r22.01 \
    --repoagent=checksum

