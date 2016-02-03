// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#ifndef CAFFE_DISTRI_MEMORYINPUTADAPTER_HPP_
#define CAFFE_DISTRI_MEMORYINPUTADAPTER_HPP_

#include "caffe/layers/memory_data_layer.hpp"
#include "util/InputAdapter.hpp"

template<typename Dtype>
class MemoryInputAdapter : public InputAdapter<Dtype> {
  protected:
    int batchSize_;
    MemoryDataLayer<Dtype>* mem_;

  public:
    MemoryInputAdapter(shared_ptr<Layer<Dtype> >& layer, int solver_mode) ;
    virtual ~MemoryInputAdapter();

    virtual void feed(vector< Blob<Dtype>* >&  dataBlobs, Dtype* labelBlobPtr);
};

#endif
