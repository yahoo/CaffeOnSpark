#!/bin/bash
# Script called by Travis to build and test Caffe.
# Travis CI tests are CPU-only for lack of compatible hardware.

set -e

if ! $WITH_CUDA; then
export CPU_ONLY=1
fi

if $WITH_IO; then
export USE_LMDB=1
export USE_LEVELDB=1
export USE_OPENCV=1
fi

make clean
make build