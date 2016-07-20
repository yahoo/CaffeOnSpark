// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#ifndef CAFFE_DISTRI_COSINPUTADAPTER_HPP_
#define CAFFE_DISTRI_COSINPUTADAPTER_HPP_

#include "util/InputAdapter.hpp"


template<typename Dtype>
class CoSInputAdapter : public InputAdapter<Dtype> {
  protected:
    int numDataTops_;
    CoSDataLayer<Dtype>* cos_;

  public:
    CoSInputAdapter(shared_ptr<Layer<Dtype> >& layer, int solver_mode);
    virtual ~CoSInputAdapter();

    virtual void feed(vector< Blob<Dtype>* >&  dataBlobs);
};

#endif
