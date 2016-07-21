// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#include "util/CoSInputAdapter.hpp"

#include <glog/logging.h>

template<typename Dtype>
CoSInputAdapter<Dtype>::CoSInputAdapter(shared_ptr<Layer<Dtype> >& layer, int solver_mode)
    : InputAdapter<Dtype>(solver_mode) {

    cos_ = (CoSDataLayer<Dtype>*)layer.get();
    LOG(INFO) << "CoSInputAdapter is used";
}

template<typename Dtype>
CoSInputAdapter<Dtype>::~CoSInputAdapter() {
    cos_ = NULL;

}

template<typename Dtype>
void CoSInputAdapter<Dtype>::feed(vector< Blob<Dtype>* >&  dataBlobs) {
    int vectlen = dataBlobs.size();
    vector<Dtype*> dataPtrs(vectlen);
    for (int i = 0; i < vectlen; i++) {
        if (InputAdapter<Dtype>::solver_mode_ == Caffe::CPU)
            dataPtrs[i] = dataBlobs[i]->mutable_cpu_data();
        else
            dataPtrs[i] = dataBlobs[i]->mutable_gpu_data();

    }
    cos_->Reset(dataPtrs);
}

REGISTER_INPUT_ADAPTER("CoSData", CoSInputAdapter);
