#!/bin/bash

~/triton/server/bin/tritonserver \
    --backend-directory=/home/ubuntu/triton/server/backends \
    --model-repository ~/models \
    --allow-http 1 \
    --http-port 8000

