// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#include <glog/logging.h>
#include <stdio.h>

#include <sstream>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "caffe/parallel.hpp"
#include "util/parallel_cpu.hpp"

namespace caffe {

enum Op {
  copy,
  replace_cpu,
  replace_gpu,
  replace_cpu_diff,
  replace_gpu_diff
};

template<typename Dtype>
static void apply_buffers(const vector<Blob<Dtype>*>& blobs,
                          Dtype* buffer, size_t total_size, Op op) {
  Dtype* ptr = buffer;
  for (int i = 0; i < blobs.size(); ++i) {
    int size = blobs[i]->count();
    switch (op) {
      case copy: {
        // Init buffer to current values of blobs
        caffe_copy(size,
                   reinterpret_cast<const Dtype*>(blobs[i]->data()->cpu_data()),
                   ptr);
        break;
      }
      case replace_cpu:
        blobs[i]->data()->set_cpu_data(ptr);
        break;
      case replace_gpu:
        blobs[i]->data()->set_gpu_data(ptr);
        break;
      case replace_cpu_diff:
        blobs[i]->diff()->set_cpu_data(ptr);
        break;
      case replace_gpu_diff:
        blobs[i]->diff()->set_gpu_data(ptr);
        break;
    }
    ptr += size;
  }
  // total_size is at least one byte
  CHECK_EQ(total_size, (ptr == buffer ? 1 : ptr - buffer));
}

// Buffer size necessary to store given blobs
template<typename Dtype>
static size_t total_size(const vector<Blob<Dtype>*>& params) {
  size_t size = 0;
  for (int i = 0; i < params.size(); ++i)
    size += params[i]->count();
  // Size have at least one byte, otherwise cudaMalloc fails if net has no
  // learnable parameters.
  return (size > 0) ? size : 1;
}

template<typename Dtype>
CPUParams<Dtype>::CPUParams(shared_ptr<Solver<Dtype> > root_solver, int device)
    : Params<Dtype>(root_solver) {

  data_ = (Dtype*) malloc(size_ * sizeof(Dtype));
  // Copy blob values
  const vector<Blob<Dtype>*>& net =
    root_solver->net()->learnable_params();
  apply_buffers(net, data_, size_, copy);

  diff_ = (Dtype*) malloc(size_ * sizeof(Dtype));
  caffe_set(size_, Dtype(0), diff_);
}

template<typename Dtype>
CPUParams<Dtype>::~CPUParams() {
  free(data_);
  free(diff_);
}

template<typename Dtype>
void CPUParams<Dtype>::configure(Solver<Dtype>* solver) const {
  const vector<Blob<Dtype>*>& net =
      solver->net()->learnable_params();

  apply_buffers(net, data_, size_, replace_cpu);
  apply_buffers(net, diff_, size_, replace_cpu_diff);
}

template<typename Dtype>
P2PSyncCPU<Dtype>::P2PSyncCPU(shared_ptr<Solver<Dtype> > root_solver,
                        P2PSyncCPU<Dtype>* parent, const SolverParameter& param)
    : CPUParams<Dtype>(root_solver, param.device_id()),
      initial_iter_(root_solver->iter()),
      solver_() {

  solver_ = root_solver;
  this->configure(solver_.get());
  solver_->add_callback(this);
}

template<typename Dtype>
P2PSyncCPU<Dtype>::~P2PSyncCPU() {
}

template<typename Dtype>
void P2PSyncCPU<Dtype>::on_start() {
}

template<typename Dtype>
void P2PSyncCPU<Dtype>::on_gradients_ready() {
  caffe_cpu_scale(size_, Dtype(1.0 / Caffe::solver_count()), diff_, diff_);
}

INSTANTIATE_CLASS(Params);
INSTANTIATE_CLASS(CPUParams);
INSTANTIATE_CLASS(P2PSyncCPU);

}  // namespace caffe
