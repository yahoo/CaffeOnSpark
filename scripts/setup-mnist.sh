#!/usr/bin/env bash
# Copyright 2016 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.
#
# This script setup mnist data into lmdb/leveldb format in ${CAFFE_ON_SPARK}/data

cd caffe-public/
wget https://raw.githubusercontent.com/BVLC/caffe/master/data/mnist/get_mnist.sh
chmod +x get_mnist.sh
./get_mnist.sh
mv *-idx*-ubyte data/mnist/
examples/mnist/create_mnist.sh
rm -rf ../data/mnist_train_lmdb
mv examples/mnist/mnist_train_lmdb ../data
rm -rf ../data/mnist_test_lmdb
mv examples/mnist/mnist_test_lmdb ../data
cd ..
