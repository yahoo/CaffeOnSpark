// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#ifdef INFINIBAND
#include <vector>
#include "boost/algorithm/string.hpp"
#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "util/rdma_sync.hpp"

namespace caffe {

template<typename Dtype>
RDMASync<Dtype>::RDMASync(shared_ptr<Solver<Dtype> > root_solver,
                          const vector<shared_ptr<RDMAChannel> >& peers,
                          int rank)
    : P2PSync<Dtype>(root_solver, NULL, root_solver->param()),
      adapter_(rank == 0 ? peers[1]->adapter() : peers[0]->adapter()),
      peers_(peers),
      rank_(rank),
      data_send_(peers.size()),
      data_recv_(peers.size()),
      diff_send_(peers.size()),
      diff_recv_(peers.size()) {

#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  CUDA_CHECK(cudaSetDevice(root_solver->param().device_id()));

  chunk(rank_, &own_offs_, &own_size_);

  for (int peer = 0; peer < peers_.size(); ++peer) {
    if (peer == rank_) {
      // Chunk for which we are master, connected to all peers. Loops must be
      // imbricated to have buffers created in the same order on all boxes.
      for (int i = 0; i < peers_.size(); ++i) {
        if (i != rank_) {
          CreateMasterBuffers(i);
        }
      }
    } else {
      // Other chunks are connected to their respective masters
      CreateWorkerBuffers(peer);
    }
  }

  CUDA_CHECK(cudaSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
void RDMASync<Dtype>::chunk(int peer, size_t* offs, size_t* size) {
  // TODO align chunks to page size?
  size_t start = (peer + 0) * size_ / peers_.size();
  size_t until = (peer + 1) * size_ / peers_.size();
  *offs = start;
  *size = until - start;
}

template<typename Dtype>
void RDMASync<Dtype>::CreateMasterBuffers(int peer) {
  RDMAChannel* channel = peers_[peer].get();
  size_t size = own_size_ * sizeof(Dtype);

  // Send data from local (rank_) to remote (peer)
  uint8_t* data = reinterpret_cast<uint8_t*>(data_ + own_offs_);
  data_send_[peer].reset(new RDMABuffer(channel, data, size));

  // Recv diff from remote (peer) to local (rank_)
  uint8_t* buffer;
  CUDA_CHECK(cudaMalloc(&buffer, size));
  diff_recv_[peer].reset(new RDMABuffer(channel, buffer, size));
}

template<typename Dtype>
void RDMASync<Dtype>::CreateWorkerBuffers(int peer) {
  RDMAChannel* channel = peers_[peer].get();
  size_t offs, size;
  chunk(peer, &offs, &size);
  size *= sizeof(Dtype);

  // Recv data from remote (peer) to local (rank_)
  uint8_t* data = reinterpret_cast<uint8_t*>(data_ + offs);
  data_recv_[peer].reset(new RDMABuffer(channel, data, size));

  // Send diff from local (rank_) to remote (peer)
  uint8_t* diff = reinterpret_cast<uint8_t*>(diff_ + offs);
  diff_send_[peer].reset(new RDMABuffer(channel, diff, size));
}

template<typename Dtype>
RDMASync<Dtype>::~RDMASync() {
  for (int i = 0; i < peers_.size(); ++i) {
    if (i != rank_) {
      CUDA_CHECK(cudaFree(diff_recv_[i]->addr()));
    }
  }
}

template<typename Dtype>
void RDMASync<Dtype>::on_start() {
  // Send weights to each node
  sync();

  // Send weights to local GPUs
  P2PSync<Dtype>::on_start();
}

template<typename Dtype>
void RDMASync<Dtype>::on_gradients_ready() {
  // Reduce gradients from local GPUs
  P2PSync<Dtype>::on_gradients_ready();

  // Send gradients to corresponding parameter server node
  int peer = rank_ + 1;
  for (int n = 0; n < peers_.size() - 1; ++n) {
    if (peer == peers_.size()) {
      peer = 0;
    }
    diff_send_[peer]->Write();
    peer++;
  }

  // Sum gradients as they are received
  for (int n = 0; n < peers_.size() - 1; ++n) {
    RDMABuffer* buffer = adapter_.received().pop();
#ifdef DEBUG
    bool ok = false;
    for (int i = 0; i < diff_recv_.size(); ++i) {
      if (buffer == diff_recv_[i].get()) {
        ok = true;
      }
    }
    CHECK(ok);
    CHECK(buffer->size() == own_size_ * sizeof(Dtype));
#endif
    Dtype* src = reinterpret_cast<Dtype*>(buffer->addr());
    Dtype* dst = diff_ + own_offs_;
    caffe_gpu_add(own_size_, src, dst, dst);
  }
}

template<typename Dtype>
void RDMASync<Dtype>::sync() {
  // Send weights to each peer
  int peer = rank_ + 1;  // To avoid all sending to same peer at the same time
  for (int n = 0; n < peers_.size() - 1; ++n) {
    if (peer == peers_.size()) {
      peer = 0;
    }
    data_send_[peer]->Write();
    peer++;
  }

  for (int n = 0; n < peers_.size() - 1; ++n) {
#ifdef DEBUG
    RDMABuffer* buffer = adapter_.received().pop();
    bool ok = false;
    for (int i = 0; i < data_recv_.size(); ++i) {
      if (buffer == data_recv_[i].get()) {
        ok = true;
      }
    }
    CHECK(ok);
#else
    adapter_.received().pop();
#endif
  }
}

INSTANTIATE_CLASS(RDMASync);
}  // namespace caffe

#endif
