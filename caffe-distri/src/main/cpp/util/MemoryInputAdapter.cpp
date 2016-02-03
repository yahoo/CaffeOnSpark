// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#include "util/MemoryInputAdapter.hpp"

#include <glog/logging.h>

template<typename Dtype>
MemoryInputAdapter<Dtype>::MemoryInputAdapter(shared_ptr<Layer<Dtype> >& layer, int solver_mode)
  : InputAdapter<Dtype>(solver_mode) {
    mem_ = (MemoryDataLayer<Dtype>*)layer.get();

    const MemoryDataParameter mem_data_param = mem_->layer_param().memory_data_param();
    batchSize_ = mem_data_param.batch_size();
    LOG(INFO) << "MemoryInputAdapter is used";
}

template<typename Dtype>
MemoryInputAdapter<Dtype>::~MemoryInputAdapter() {
    mem_ = NULL;
}

template<typename Dtype>
void MemoryInputAdapter<Dtype>::feed(vector< Blob<Dtype>* >&  dataBlobs, Dtype* labelBlobPtr) {
    if (InputAdapter<Dtype>::solver_mode_ == Caffe::CPU)
        mem_->Reset(dataBlobs[0]->mutable_cpu_data(), labelBlobPtr, batchSize_);
    else
        mem_->Reset(dataBlobs[0]->mutable_gpu_data(), labelBlobPtr, batchSize_);
}

REGISTER_INPUT_ADAPTER("MemoryData", MemoryInputAdapter);
