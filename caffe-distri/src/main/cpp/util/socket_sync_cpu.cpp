// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#include <vector>
#include "boost/algorithm/string.hpp"
#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "util/socket_sync_cpu.hpp"


namespace caffe {

template<typename Dtype>
SocketSyncCPU<Dtype>::SocketSyncCPU(shared_ptr<Solver<Dtype> > root_solver,
                              const vector<shared_ptr<SocketChannel> > &
                              peers, int rank)
  : P2PSyncCPU<Dtype>(root_solver, NULL, root_solver->param()),
    peers_(peers),
    rank_(rank),
    data_send_(peers.size()),
    data_recv_(peers.size()),
    diff_send_(peers.size()),
    diff_recv_(peers.size()) {

  chunk(rank_, &own_offs_, &own_size_);
  for (int peer = 0; peer < peers_.size(); ++peer) {
    if (peer == rank_) {
      // Chunk for which we are master, connected to all peers.
      // Loops must be imbricated to have buffers created in
      // the same order on all boxes.
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
}

template<typename Dtype>
void SocketSyncCPU<Dtype>::chunk(int peer, size_t* offs,
                              size_t* size) {
  // TODO align chunks to page size?
  size_t start = (peer + 0) * size_ / peers_.size();
  size_t until = (peer + 1) * size_ / peers_.size();
  *offs = start;
  *size = until - start;
}

template<typename Dtype>
void SocketSyncCPU<Dtype>::CreateMasterBuffers(int peer) {
  SocketChannel* channel = peers_[peer].get();
  size_t size = own_size_ * sizeof(Dtype);

  // Send data from local (rank_) to remote (peer)
  uint8_t* data = reinterpret_cast<uint8_t*>(data_ + own_offs_);
  data_send_[peer].reset(new SocketBuffer(this->rank_, channel, data,
                                          size, NULL));

  // Recv diff from remote (peer) to local (rank_)
  uint8_t* buffer;
  buffer = reinterpret_cast<uint8_t*>(malloc(size));
  diff_recv_[peer].reset(new SocketBuffer(this->rank_, channel, NULL,
                                          size, buffer));
}

template<typename Dtype>
void SocketSyncCPU<Dtype>::CreateWorkerBuffers(int peer) {
  SocketChannel* channel = peers_[peer].get();
  size_t offs, size;
  chunk(peer, &offs, &size);
  size *= sizeof(Dtype);

  // Recv data from remote (peer) to local (rank_)
  uint8_t* data = reinterpret_cast<uint8_t*>(data_ + offs);
  data_recv_[peer].reset(new SocketBuffer(this->rank_, channel, NULL,
                                          size, data));

  // Send diff from local (rank_) to remote (peer)
  uint8_t* diff = reinterpret_cast<uint8_t*>(diff_ + offs);
  diff_send_[peer].reset(new SocketBuffer(this->rank_, channel,
                                          diff, size, NULL));
}

template<typename Dtype>
SocketSyncCPU<Dtype>::~SocketSyncCPU() {
  for (int i = 0; i < peers_.size(); ++i) {
    if (i != rank_) {
      free(diff_recv_[i]->addr());
    }
  }
}

template<typename Dtype>
void SocketSyncCPU<Dtype>::on_start() {
  // Send weights to each node
  sync();
}

template<typename Dtype>
void SocketSyncCPU<Dtype>::on_gradients_ready() {
  // Reduce gradients from local CPU.
  P2PSyncCPU<Dtype>::on_gradients_ready();
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
  peer = rank_ + 1;
  for (int n = 0; n < peers_.size() - 1; ++n) {
    if (peer == peers_.size()) {
      peer = 0;
    }
    SocketBuffer * buffer = diff_recv_[peer]->Read();
    Dtype* src = reinterpret_cast<Dtype*>(buffer->addr());
    Dtype* dst = diff_ + own_offs_;
    caffe_add(own_size_, src, dst, dst);
    peer++;
  }
}

template<typename Dtype>
void SocketSyncCPU<Dtype>::sync() {
  // Send weights to each peer
  int peer = rank_ + 1;  // To avoid all sending to same peer at
  // the same time
  for (int n = 0; n < peers_.size() - 1; ++n) {
    if (peer == peers_.size()) {
      peer = 0;
    }
    data_send_[peer]->Write();
    peer++;
  }

  peer = rank_ + 1;
  for (int n = 0; n < peers_.size() - 1; ++n) {
    if (peer == peers_.size()) {
      peer = 0;
    }
    data_recv_[peer]->Read();
    peer++;
  }
}

INSTANTIATE_CLASS(SocketSyncCPU);
}  // namespace caffe
