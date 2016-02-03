// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#ifndef CPU_ONLY
#include "caffe/parallel.hpp"
#include "util/socket_sync.hpp"
#else
#include "util/parallel_cpu.hpp"
#include "util/socket_sync_cpu.hpp"
#endif
#include "util/mini_cluster.hpp"
#include "util/socket.hpp"
#ifdef INFINIBAND
#include "util/rdma_sync.hpp"
#endif
namespace caffe {

template<typename Dtype>
MiniCluster<Dtype>::MiniCluster(const string& host, int size)
    : size_(size),
      sockets_() {
  bool server = host.size() == 0;
  if (server) {
    // Server gives itself rank 0
    rank_ = 0;
    Socket server(host, PORT, true);
    for (int rank = 1; rank < size; ++rank) {
      shared_ptr<Socket> socket = server.accept();
      LOG(INFO)<< "Client " << rank << "/" << (size - 1) << " connected\n";
      sockets_.push_back(socket);
      socket->writeInt(rank);
    }
  } else {
    // Then clients get their rank by connection order
    shared_ptr<Socket> socket(new Socket(host, PORT, false));
    sockets_.push_back(socket);
    rank_ = socket->readInt();
  }
}

template<typename Dtype>
void MiniCluster<Dtype>::AllGather(vector<string>* data) {
  CHECK_EQ(data->size(), 1);
  if (rank_ == 0) {
    // Server listens to each client's data and then send vector
    for (size_t i = 0; i < sockets_.size(); ++i) {
      data->push_back(sockets_[i]->readStr());
    }
    for (size_t i = 0; i < sockets_.size(); ++i) {
      for (size_t j = 0; j < data->size(); ++j) {
        sockets_[i]->writeStr(data->at(j));
      }
    }
  } else {
    // Clients send their data and wait for vector
    sockets_[0]->writeStr(data->at(0));
    data->clear();
    for (int i = 0; i < size_; ++i) {
      data->push_back(sockets_[0]->readStr());
    }
  }
}

template<typename Dtype>
void MiniCluster<Dtype>::run(shared_ptr<Solver<Dtype> > root_solver,
                             const vector<int>& gpus,
                             int total_gpus) {
#ifdef INFINIBAND
  RDMAAdapter adapter;
  LOG(INFO) << "Found RDMA adapter " << adapter.name();

  // Create channel for each peer
  vector<shared_ptr<RDMAChannel> > peers(size_);
  for (int i = 0; i < size_; ++i) {
    if (i != rank_) {
      peers[i].reset(new RDMAChannel(adapter));
    }
  }
  // Connect channels all to all
  for (int i = 0; i < size_; ++i) {
    vector<string> addresses(1);
    if (i != rank_) {
      addresses[0] = peers[i]->address();
    }
    AllGather(&addresses);
    for (int j = 0; j < addresses.size(); ++j)
      LOG(INFO) << addresses[j];
    if (i == rank_) {
      for (int j = 0; j < size_; ++j) {
        if (j != rank_) {
          peers[j]->Connect(addresses[j]);
        }
      }
    }
  }
  vector<shared_ptr<P2PSync<Dtype> > > syncs(gpus.size());
  // RDMASync will create all necessary buffers
  syncs[0].reset(new RDMASync<Dtype>(root_solver, peers, rank_));
#else
  // Create channel for each peer
  vector<shared_ptr<SocketChannel> > peers(size_);
  for (int i = 0; i < size_; ++i) {
    if (i != rank_) {
      peers[i].reset(new SocketChannel());
    }
  }

  SocketAdapter adapter(&peers);
  usleep(10000);
  // Get all channels to connect to
  vector<string> addresses(1);
  // Set local address to send to master in AllGather.
  // If you are master, you still need to set it, so
  // that it is sent to everyone during regular broadcast in AllGather
  addresses[0] = adapter.address();
  LOG(INFO) << "Adapter address " << adapter.address().c_str();
  AllGather(&addresses);
  for (int j = 0; j < addresses.size(); ++j)
    LOG(INFO) << "ADDRESS [" << addresses.at(j).c_str() << "]";

  // Connect to all channnels
  for (int j = 0; j < size_; ++j) {
    if (j != rank_) {
      LOG(INFO) << "Connecting to [" << addresses[j].c_str() << "]";
      peers[j]->Connect(addresses[j]);
    }
  }

#ifndef CPU_ONLY
  vector<shared_ptr<P2PSync<Dtype> > > syncs(gpus.size());
  syncs[0].reset(new SocketSync<Dtype>(root_solver, peers, rank_));
#else
  vector<shared_ptr<P2PSyncCPU<Dtype> > > syncs(1);
  syncs[0].reset(new SocketSyncCPU<Dtype>(root_solver, peers, rank_));
#endif
#endif

#ifndef CPU_ONLY
  syncs[0]->prepare(gpus, &syncs);
  LOG(INFO)<< "Starting Optimization";

  // Switch to total number of GPUs once the datareaders are ready
  Caffe::set_solver_count(total_gpus);
  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StartInternalThread();
  }

  // Run root solver on current thread
  syncs[0]->solver()->Solve();

  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StopInternalThread();
  }
#else
  Caffe::set_solver_count(1);
  LOG(INFO) << "Starting solver...";
  syncs[0]->solver()->Solve();
#endif

}

INSTANTIATE_CLASS(MiniCluster);

}  // namespace caffe

