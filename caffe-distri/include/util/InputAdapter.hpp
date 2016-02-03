// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#ifndef CAFFE_DISTRI_INPUTADAPTER_HPP_
#define CAFFE_DISTRI_INPUTADAPTER_HPP_

#include <vector>
#include <map>

#include "common.hpp"
#include "caffe/caffe.hpp"

template<typename Dtype>
class InputAdapter {
  protected:
    int solver_mode_;

  public:
    /**
     * Create a network data layer to be used by feedNet()
     */
    InputAdapter(int solver_mode) {
        solver_mode_ = solver_mode;
    }
    virtual ~InputAdapter() {}

    /**
     * feed a layer with data and label
     */
    virtual void feed(vector< Blob<Dtype>* >&  dataBlobs, Dtype* labelPtr) = 0;
};

struct cmp_str
{
   bool operator()(const char *a, const char *b) const {
      return std::strcmp(a, b) < 0;
   }
};

template<typename Dtype>
class InputAdapterRegistry {
  public:
    typedef InputAdapter<Dtype>* (*AdapterCreator)(shared_ptr<Layer<Dtype> >& layer, int solver_mode);
    typedef map<const char *, AdapterCreator, cmp_str> CreatorRegistry;

    static void Set(const char* layertypename, AdapterCreator factory);
    static InputAdapter<Dtype>* MakeAdapter(shared_ptr<Layer<Dtype> >& layer, int solver_mode);

  private:
    // AdapterFactory registry should never be instantiated - everything is done with its
    // static variables.
    InputAdapterRegistry() {}

    static CreatorRegistry& Registry() {
       static CreatorRegistry* g_registry_ = new CreatorRegistry();
       return *g_registry_;
    }
};

template <typename Dtype>
class InputAdapterRegisterer {
 public:
   InputAdapterRegisterer(const char * layerNm,
                         InputAdapter<Dtype>* (*creator)(shared_ptr<Layer<Dtype> >& layer, int solver_mode)) {
     InputAdapterRegistry<Dtype>::Set(layerNm, creator);
   }
};

#define REGISTER_INPUT_ADAPTER(layerNm, adapter) \
  template<typename Dtype> \
  InputAdapter<Dtype>* adapter##Creator(shared_ptr<Layer<Dtype> >& layer, int solver_mode) { \
     return new adapter<Dtype>(layer, solver_mode); \
  }; \
  static InputAdapterRegisterer<float> g_f_creator_##adapter(layerNm, adapter##Creator<float>);    \
  static InputAdapterRegisterer<double> g_d_creator_##adapter(layerNm, adapter##Creator<double>)

#endif
