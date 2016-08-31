// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#ifndef CAFFE_DISTRI_RDMA_HPP_
#define CAFFE_DISTRI_RDMA_HPP_

#ifdef INFINIBAND
#include <infiniband/verbs.h>

#include <boost/thread.hpp>
#include <string>
#include <vector>

#include "caffe/common.hpp"

// a large number to distinguish ctrl buffer ids from data buffer ids.
#define CTRL_ID_OFFSET (1<<16)

/**
 * Minimal wrapper on top of ibverbs. We avoided MPI for now to simplify
 * integration in Spark, and potential future ones in BIDMach and
 * IPython.parallel. This API currently supports point to point transfers
 * using RDMA Write between pre-allocated buffers. Buffers can reside on CPU
 * or GPU memory, and a notification is raised on the receiving end when a
 * buffer has been updated.
 */

namespace caffe {

class RDMABuffer;

/**
 * A single instance of this class is typically created during process start,
 * and destroyed on exit. From this adapter, several channels can be created
 * toward remote processes. Each channel can be then be used to create several
 * buffers.
 */
class RDMAAdapter : InternalThread {
 public:
  RDMAAdapter();
  ~RDMAAdapter();

  // Adapter name, e.g. mlx5_0.
  string name() const;

  /**
   * When a buffer gets updated by a remote write, it will be put in this
   * queue on the receiving side as a notification.
   */
  BlockingQueue<RDMABuffer*>& received() const {
    return received_;
  }

  BlockingQueue<RDMABuffer*>& ctrl_received() const {
    return ctrl_received_;
  }

 protected:
  void InternalThreadEntry();

  mutable BlockingQueue<RDMABuffer*> received_;

  mutable BlockingQueue<RDMABuffer*> ctrl_received_;

  static const int MAX_CONCURRENT_WRITES = 256;

  ibv_context* context_;
  // ibverbs protection domain
  ibv_pd* pd_;
  // Completion event channel, to wait for work completions
  ibv_comp_channel* channel_;
  // Completion queue, to poll on work completions
  ibv_cq* cq_;
  // Pre-allocated work completions array used for polling
  ibv_wc wc_[MAX_CONCURRENT_WRITES * 2];

  friend class RDMABuffer;
  friend class RDMAChannel;

DISABLE_COPY_AND_ASSIGN(RDMAAdapter);
};

/**
 * Point to point bidirectional communication with a peer. Multiple channels
 * can be created by process, and connected together by exchanging addresses.
 */
class RDMAChannel {
 public:
  explicit RDMAChannel(const RDMAAdapter& adapter);
  ~RDMAChannel();

  const RDMAAdapter& adapter() {
    return adapter_;
  }

  /**
   * Address must be exchanged between hosts, e.g. though TCP or some other
   * mechanism to allow connection of each ends of a channel. Both ends
   * addresses must be exchanged before creating buffers.
   */
  string address() const;
  void Connect(const string& address);

  mutable boost::mutex mutex_;

 protected:
  const RDMAAdapter& adapter_;
  mutable vector<RDMABuffer*> buffers_;

  static const int MAX_BUFFERS = 64;

  // Allows exchanging memory regions representing buffers over the channel,
  // instead of requiring user to exchange manually like channel addresses.
  mutable vector<ibv_mr*> memory_regions_;
  mutable vector<ibv_mr*> region_regions_;
  mutable int memory_regions_received_;
  BlockingQueue<ibv_mr*> memory_regions_queue_;
  void SendMR(ibv_mr* mr, int id);
  void RecvMR(int id);

  void recv();

  class Address {
   public:
    uint32_t lid;
    uint32_t qpn;
    uint32_t psn;
  };
  Address self_;
  ibv_cq* write_cq_;
  ibv_qp* qp_;

  friend class RDMAAdapter;
  friend class RDMABuffer;

DISABLE_COPY_AND_ASSIGN(RDMAChannel);
};

/**
 * Once a channel has been established between two boxes, buffers can be
 * created on each side. Buffers are paired in a synchronous way, the
 * constructor blocks until the other side has reached the same point. Once it
 * exits, and the buffer has been created, it is mapped to the remote one
 * and is ready to be used.
 */
class RDMABuffer {
 public:
  RDMABuffer(RDMAChannel* channel, uint8_t* addr, size_t size);
  virtual ~RDMABuffer();

  uint8_t* addr() const {
    return addr_;
  }
  const size_t size() const {
    return size_;
  }

  // Asynchronously writes content to remote peer
  void Write(bool data=true);

 protected:
  RDMAChannel* channel_;
  uint8_t* addr_;
  const size_t size_;

  ibv_mr* self_;
  ibv_mr* peer_;
  int id_;

  friend class RDMAAdapter;

DISABLE_COPY_AND_ASSIGN(RDMABuffer);
};

}  // namespace caffe

#endif
#endif
