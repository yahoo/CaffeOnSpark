// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#ifndef CAFFE_DISTRI_SOCKET_SYNC_HPP_
#define CAFFE_DISTRI_SOCKET_SYNC_HPP_

#include <boost/shared_ptr.hpp>
#include <vector>
#include "caffe/parallel.hpp"
#include "caffe/solver.hpp"
#include "util/socket.hpp"

namespace caffe {
/**
 * Synchronous data parallelism between machines over RDMA. It builds on top
 * of the existing single node multi-GPU code in Caffe, by adding an extra
 * step to synchronize nodes' root GPUs.
 *
 * During creation, the weight and gradient buffers are sharded by the number
 * of nodes in the cluster. Each node is assigned a shard for which it will
 * behave as a parameter server. All nodes contain and compute on the full
 * buffers, but are only parameter servers for a subset.
 *
 * An SGD iteration goes as follow, first each node sends its shard of weights
 * to all others. This could be implemented using a broadcast collective, but
 * since all nodes send to all others concurrently, bandwidth is uniform, and
 * point to point communication should already be optimal.
 *
 * Each node's root GPU now has the weights ready, and propagates them to other
 * GPUs using Caffe's single node code. Gradients are then computed using a
 * forward/backward pass on each GPU, and reduced to root GPUs, again using
 * the single node code.
 *
 * The last step is symmetric to the first, gradients are sharded, and each
 * node sends their shards to their respective parameter server peer. Transfers
 * are again concurrent, and bandwidth uniform between nodes. Each node then
 * averages gradients for which it is parameter server, and applies the solver.
 * The solver code has not been optimized to run only on the relevant shard,
 * the remaining weights are simply ignored and will be overridden during the
 * first phase of the next iteration.
 */
template<typename Dtype>
class SocketSync : public P2PSync<Dtype> {
 public:
  SocketSync(shared_ptr<Solver<Dtype> > solver,
             const vector<shared_ptr<SocketChannel> >& peers, int rank);
  virtual ~SocketSync();
  void sync();

 protected:
  void chunk(int peer, size_t* offs, size_t* size);
  void CreateMasterBuffers(int peer);
  void CreateWorkerBuffers(int peer);

  virtual void on_start();
  virtual void on_gradients_ready();

  vector<shared_ptr<SocketChannel> > peers_;
  // Rank of the current node, MPI like
  int rank_;
  // Each node is parameter server for a shard, defined as an offset and size
  size_t own_offs_;
  size_t own_size_;
  // RDMA mappings on weights and gradients buffers, allow send and receive
  vector<shared_ptr<SocketBuffer> > data_send_;
  vector<shared_ptr<SocketBuffer> > data_recv_;
  vector<shared_ptr<SocketBuffer> > diff_send_;
  vector<shared_ptr<SocketBuffer> > diff_recv_;

  // Weights and gradients buffers and size
  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
  DISABLE_COPY_AND_ASSIGN(SocketSync);
};

}  // namespace caffe

#endif
