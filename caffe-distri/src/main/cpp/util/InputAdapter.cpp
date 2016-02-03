// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#include "util/InputAdapter.hpp"

#include <glog/logging.h>

template<typename Dtype>
void InputAdapterRegistry<Dtype>::Set(const char* layertypename, AdapterCreator creator) {
   CreatorRegistry& registry = Registry();
   registry[layertypename] = creator;
}

template<typename Dtype>
InputAdapter<Dtype>* InputAdapterRegistry<Dtype>::MakeAdapter(shared_ptr<Layer<Dtype> >& layer, int solver_mode) {
   typename CreatorRegistry::iterator it;

   CreatorRegistry& registry = Registry();
   it = registry.find(layer->type());
   if (it != registry.end())
      return (it->second)(layer, solver_mode);
   LOG(WARNING) << "Failed to find adapter creator for layer type \'" << layer->type() << "\'";
   return NULL;
}

INSTANTIATE_CLASS(InputAdapterRegistry);

