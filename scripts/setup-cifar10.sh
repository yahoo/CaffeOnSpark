#!/usr/bin/env bash
# Copyright 2016 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.
#
# This script setup cifar data into lmdb format in ${CAFFE_ON_SPARK}/data

pushd ${CAFFE_ON_SPARK}/caffe-public/
./data/cifar10/get_cifar10.sh
./examples/cifar10/create_cifar10.sh
rm -rf ../data/cifar10_train_lmdb
mv examples/cifar10/cifar10_train_lmdb ../data
rm -rf ../data/cifar10_test_lmdb
mv examples/cifar10/cifar10_test_lmdb ../data
rm -f ../data/mean.binaryproto
mv examples/cifar10/mean.binaryproto ../data
popd
