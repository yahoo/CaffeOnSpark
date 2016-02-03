// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#ifndef CAFFE_PARALLEL_CPU_HPP_
#define CAFFE_PARALLEL_CPU_HPP_

#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

// Params stored in CPU memory.
template<typename Dtype>
class CPUParams : public Params<Dtype> {
 public:
  CPUParams(shared_ptr<Solver<Dtype> > root_solver, int device);
  virtual ~CPUParams();

  void configure(Solver<Dtype>* solver) const;

 protected:
  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

// Synchronous data parallelism using map-reduce between local CPUs.
template<typename Dtype>
class P2PSyncCPU : public CPUParams<Dtype>, public Solver<Dtype>::Callback {
 public:
  explicit P2PSyncCPU(shared_ptr<Solver<Dtype> > root_solver,
                   P2PSyncCPU<Dtype>* parent, const SolverParameter& param);
  virtual ~P2PSyncCPU();

  inline const shared_ptr<Solver<Dtype> >& solver() const {
    return solver_;
  }

  inline const int GetInitIter() const { return initial_iter_; }

 protected:
  void on_start();
  void on_gradients_ready();

  const int initial_iter_;
  shared_ptr<Solver<Dtype> > solver_;

  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

}  // namespace caffe

#endif
