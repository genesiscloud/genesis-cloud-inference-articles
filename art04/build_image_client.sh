#!/bin/bash

mkdir -p ./bin

CLI_INC=~/triton/client/include
CLI_LIB=~/triton/client/lib

g++ -o ./bin/image_client --std=c++11 \
    -I $CLI_INC \
    image_client.cpp \
    -L $CLI_LIB -lhttpclient


