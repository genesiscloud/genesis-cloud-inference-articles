#!/bin/bash

cmake \
    -DCMAKE_INSTALL_PREFIX=/home/ubuntu/triton/client \
    -DTRITON_ENABLE_CC_HTTP=ON \
    -DTRITON_ENABLE_CC_GRPC=ON \
    -DTRITON_ENABLE_PERF_ANALYZER=ON \
    -DTRITON_ENABLE_PYTHON_HTTP=ON \
    -DTRITON_ENABLE_PYTHON_GRPC=ON \
    -DTRITON_ENABLE_JAVA_HTTP=OFF \
    -DTRITON_ENABLE_GPU=ON \
    -DTRITON_ENABLE_EXAMPLES=ON \
    -DTRITON_ENABLE_TESTS=ON \
    -DTRITON_COMMON_REPO_TAG=r22.01 \
    -DTRITON_THIRD_PARTY_REPO_TAG=r22.01 \
    -DTRITON_CORE_REPO_TAG=r22.01 \
    -DTRITON_BACKEND_REPO_TAG=r22.01 \
    ..

