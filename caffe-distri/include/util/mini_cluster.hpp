// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#ifndef CAFFE_DISTRI_MINI_CLUSTER_HPP_
#define CAFFE_DISTRI_MINI_CLUSTER_HPP_

#include <boost/shared_ptr.hpp>
#include <string>
#include <vector>

#include "caffe/solver.hpp"
#include "util/socket.hpp"

namespace caffe {
/**
 * Mini clustering solution using sockets for standalone Caffe. It is currently
 * only used to exchange RDMA addresses between nodes when the cluster starts,
 * but could be extended to exchange data as well if RDMA is not available.
 *
 * When running in Spark, RDMA addresses can be exchanged through a distributed
 * task, so MiniCluster has been used mostly for debugging purposes.
 */
template<typename Dtype>
class MiniCluster {
 public:
  MiniCluster(const string& host, int size);
  // MPI style cluster size and current process rank
  int size() const {
    return size_;
  }
  int rank() const {
    return rank_;
  }
  // MPI style data exchange
  void AllGather(vector<string>* data);
  /**
   * Main method: runs Caffe training distributed over RDMA. First establishes
   * TCP connections to exchange RDMA addresses between all nodes. Then
   * creates RDMASync instances for each node's root GPU. Finally starts
   * the threads to perform the actual training.
   */
  void run(shared_ptr<Solver<Dtype> > root_solver,
           const vector<int>& gpus,
           int total_gpus);
  static const int PORT = 59923;

 protected:
  const int size_;
  int rank_;
  vector<shared_ptr<Socket> > sockets_;

  DISABLE_COPY_AND_ASSIGN(MiniCluster);
};
}  // namespace caffe

#endif
