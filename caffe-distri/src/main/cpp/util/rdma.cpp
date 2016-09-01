// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#ifdef INFINIBAND
#include <glog/logging.h>
#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#include "util/rdma.hpp"

namespace caffe {

ibv_context* open_default_device() {
  ibv_device** dev_list;
  ibv_device* ib_dev;
  dev_list = ibv_get_device_list(NULL);
  CHECK(dev_list) << "No InfiniBand device found";
  ib_dev = dev_list[0];
  CHECK(ib_dev) << "No InfiniBand device found";
  ibv_context* context = ibv_open_device(ib_dev);
  CHECK(context) << "Open context failed for " << ibv_get_device_name(ib_dev);
  return context;
}

ibv_pd* alloc_protection_domain(ibv_context* context) {
  ibv_pd* pd = ibv_alloc_pd(context);
  CHECK(pd) << "Failed to allocate protection domain";
  return pd;
}

RDMAAdapter::RDMAAdapter()
    : context_(open_default_device()),
      pd_(alloc_protection_domain(context_)) {
  channel_ = ibv_create_comp_channel(context_);
  CHECK(channel_) << "Failed to create completion channel";
  cq_ = ibv_create_cq(context_, MAX_CONCURRENT_WRITES * 2, NULL, channel_, 0);
  CHECK(cq_) << "Failed to create completion queue";
  CHECK(!ibv_req_notify_cq(cq_, 0)) << "Failed to request CQ notification";

  StartInternalThread();
}

RDMAAdapter::~RDMAAdapter() {
  StopInternalThread();

  CHECK(!ibv_destroy_cq(cq_)) << "Failed to destroy CQ";
  CHECK(!ibv_destroy_comp_channel(channel_)) << "Failed to destroy channel";
  CHECK(!ibv_dealloc_pd(pd_)) << "Failed to deallocate PD";
  CHECK(!ibv_close_device(context_)) << "Failed to release context";
}

string RDMAAdapter::name() const {
  return string(context_->device->name);
}

/**
 * Polling for events on a inner thread allows processing of management messages
 * like buffer connection immediately, even if the user is not polling.
 * Otherwise buffer constructors would block indefinitely.
 *
 * Deep learning workloads are about sending small numbers of large messages,
 * in which case this model works great. If the library was to be used to
 * exchange large numbers of short messages, it would be useful to split
 * management and data messages over two different queue pairs. User threads
 * could then wait or poll on the data queue pair directly.
 */
void RDMAAdapter::InternalThreadEntry() {
  while (!must_stop()) {
    ibv_cq* cq;
    void* cq_context;
    CHECK(!ibv_get_cq_event(channel_, &cq, &cq_context));
    CHECK(cq == cq_);
    ibv_ack_cq_events(cq, 1);
    CHECK(!ibv_req_notify_cq(cq_, 0));

    int ne = ibv_poll_cq(cq_, MAX_CONCURRENT_WRITES * 2,
      static_cast<ibv_wc*>(wc_));
    CHECK_GE(ne, 0);

    for (int i = 0; i < ne; ++i) {
      CHECK(wc_[i].status == IBV_WC_SUCCESS) << "Failed status \n"
                                             << ibv_wc_status_str(wc_[i].status)
                                             << " " << wc_[i].status << " "
                                             << static_cast<int>(wc_[i].wr_id)
                                             << " "<< wc_[i].vendor_err;

      if (wc_[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
        // Data message, add it to user received queue
        RDMAChannel* channel = reinterpret_cast<RDMAChannel*>(wc_[i].wr_id);
        channel->recv();
        int id = wc_[i].imm_data;
        if (id >= CTRL_ID_OFFSET) {
        // ctrl signal
          ctrl_received_.push(channel->buffers_[id - CTRL_ID_OFFSET]);
        } else {
        // data
          received_.push(channel->buffers_[id]);
        }
      } else {
        if (wc_[i].opcode & IBV_WC_RECV) {
          // Buffer connection message
          RDMAChannel* channel = reinterpret_cast<RDMAChannel*>(wc_[i].wr_id);
          int id = wc_[i].imm_data;
          channel->memory_regions_queue_.push(channel->memory_regions_[id]);
          CHECK(id == channel->memory_regions_received_++);
          CHECK(!ibv_dereg_mr(channel->region_regions_[id]));
        }
      }
    }
  }
}

//

RDMAChannel::RDMAChannel(const RDMAAdapter& adapter)
    : adapter_(adapter),
      buffers_(),
      memory_regions_(MAX_BUFFERS),
      region_regions_(MAX_BUFFERS),
      memory_regions_received_() {

  // Create write completion queue
  write_cq_ = ibv_create_cq(adapter_.context_, 1, NULL, NULL, 0);
  CHECK(write_cq_) << "Failed to create completion queue";

  // Create queue pair
  {
    struct ibv_qp_init_attr attr;
    caffe_memset(sizeof(ibv_qp_init_attr), 0, &attr);
    attr.send_cq = write_cq_;
    attr.recv_cq = adapter.cq_;
    attr.cap.max_send_wr = RDMAAdapter::MAX_CONCURRENT_WRITES;
    attr.cap.max_recv_wr = RDMAAdapter::MAX_CONCURRENT_WRITES;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    attr.qp_type = IBV_QPT_RC;

    qp_ = ibv_create_qp(adapter.pd_, &attr);
    CHECK(qp_) << "Failed to create queue pair";
  }

  // Init queue pair
  {
    struct ibv_qp_attr attr;
    caffe_memset(sizeof(ibv_qp_attr), 0, &attr);
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = 1;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;

    int mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT
        | IBV_QP_ACCESS_FLAGS;
    CHECK(!ibv_modify_qp(qp_, &attr, mask)) << "Failed to set QP to INIT";
  }

  // Local address
  {
    struct ibv_port_attr attr;
    CHECK(!ibv_query_port(adapter.context_, (uint8_t) 1, &attr))
        << "Query port";
    self_.lid = attr.lid;
    self_.qpn = qp_->qp_num;
    self_.psn = caffe_rng_rand() & 0xffffff;
  }

  for (int i = 0; i < MAX_BUFFERS; ++i) {
    RecvMR(i);
  }

  // Create initial recv request for data. 
  recv();
  // Create initial recv request for ctrl signals.
  recv();
}

RDMAChannel::~RDMAChannel() {
  CHECK(!ibv_destroy_qp(qp_)) << "Failed to destroy QP";
}

// Switch to hexadecimal to simplify transfer using text-based tools
static string hex(uint8_t* data, size_t size) {
  string hex(size * 2, ' ');
  for (int i = 0; i < size; ++i) {
    snprintf(&hex[i * 2], hex.length(), "%02x", data[i]);
  }
  return hex;
}

static void hex(string hex, uint8_t* data) {
  size_t size = hex.size() / 2;
  for (int i = 0; i < size; ++i) {
    sscanf(&hex[i * 2], "%02x", reinterpret_cast<unsigned int*>(&data[i]));
  }
}

string RDMAChannel::address() const {
  uint8_t* bytes = const_cast<uint8_t*>(
                   reinterpret_cast<const uint8_t*>(&self_));
  return hex(bytes, sizeof(Address));
}

void RDMAChannel::Connect(const string& address) {
  Address peer;
  uint8_t* bytes = reinterpret_cast<uint8_t*>(&peer);
  size_t size = sizeof(Address);
  CHECK_EQ(address.size(), size * 2);
  hex(address, bytes);

  struct ibv_qp_attr attr;
  caffe_memset(sizeof(ibv_qp_attr), 0, &attr);
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = IBV_MTU_4096;
  attr.dest_qp_num = peer.qpn;
  attr.rq_psn = peer.psn;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;
  attr.ah_attr.is_global = 0;
  attr.ah_attr.dlid = peer.lid;
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = 1;

  int r;
  CHECK(!(r = ibv_modify_qp(qp_, &attr,
              IBV_QP_STATE |
              IBV_QP_AV |
              IBV_QP_PATH_MTU |
              IBV_QP_DEST_QPN |
              IBV_QP_RQ_PSN |
              IBV_QP_MAX_DEST_RD_ATOMIC |
              IBV_QP_MIN_RNR_TIMER))) << "QP to Ready to Receive " << r;

  caffe_memset(sizeof(ibv_qp_attr), 0, &attr);
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = self_.psn;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7; /* infinite */
  attr.max_rd_atomic = 1;

  CHECK(!(r = ibv_modify_qp(qp_, &attr,
              IBV_QP_STATE |
              IBV_QP_TIMEOUT |
              IBV_QP_RETRY_CNT |
              IBV_QP_RNR_RETRY |
              IBV_QP_SQ_PSN |
              IBV_QP_MAX_QP_RD_ATOMIC))) << "QP to Ready to Send " << r;
}

void RDMAChannel::recv() {
  struct ibv_recv_wr wr;
  caffe_memset(sizeof(wr), 0, &wr);
  wr.wr_id = (uint64_t) this;

  struct ibv_recv_wr* bad_wr;
  CHECK(!ibv_post_recv(qp_, &wr, &bad_wr)) << "Failed to post recv";
}

/**
 * Sends a buffer's memory region so that it can be mapped to it's remote end.
 */
void RDMAChannel::SendMR(ibv_mr* mr, int id) {
  // Map the memory region itself so that it can be sent
  ibv_mr* init = ibv_reg_mr(adapter_.pd_, mr, sizeof(ibv_mr),
                            IBV_ACCESS_LOCAL_WRITE);

  struct ibv_sge list;
  list.addr = (uint64_t) mr;
  list.length = sizeof(ibv_mr);
  list.lkey = init->lkey;

  struct ibv_send_wr wr;
  caffe_memset(sizeof(wr), 0, &wr);
  wr.wr_id = (uint64_t) init;
  wr.sg_list = &list;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = id;

  struct ibv_send_wr *bad_wr;
  CHECK(!ibv_post_send(qp_, &wr, &bad_wr));

  for (;;) {
    ibv_wc wc;
    int ne = ibv_poll_cq(write_cq_, 1, &wc);
    CHECK_GE(ne, 0);
    if (ne && wc.wr_id == (uint64_t) init) {
      break;
    }
  }
  CHECK(!ibv_dereg_mr(init));
}

void RDMAChannel::RecvMR(int id) {
  memory_regions_[id] = new ibv_mr();

  // Map the memory region itself so that it can be received
  ibv_mr* init = ibv_reg_mr(adapter_.pd_, memory_regions_[id], sizeof(ibv_mr),
                            IBV_ACCESS_LOCAL_WRITE);
  region_regions_[id] = init;

  struct ibv_sge list;
  list.addr = (uint64_t) memory_regions_[id];
  list.length = sizeof(ibv_mr);
  list.lkey = init->lkey;

  struct ibv_recv_wr wr;
  caffe_memset(sizeof(wr), 0, &wr);
  wr.wr_id = (uint64_t) this;
  wr.sg_list = &list;
  wr.num_sge = 1;

  struct ibv_recv_wr* bad_wr;
  CHECK(!ibv_post_recv(qp_, &wr, &bad_wr));
}

//

RDMABuffer::RDMABuffer(RDMAChannel* channel, uint8_t* addr, size_t size)
    : channel_(channel),
      addr_(addr),
      size_(size) {
  self_ = ibv_reg_mr(channel_->adapter_.pd_, addr, size,
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  CHECK(self_) << "Failed to register memory region";

  id_ = channel_->buffers_.size();
  channel_->buffers_.push_back(this);

  channel_->SendMR(self_, id_);
  peer_ = channel_->memory_regions_queue_.pop();
}

RDMABuffer::~RDMABuffer() {
  CHECK(!ibv_dereg_mr(self_));
}

void RDMABuffer::Write(bool data) {
  struct ibv_sge list;
  list.addr = (uint64_t) addr_;
  list.length = size_;
  list.lkey = self_->lkey;

  struct ibv_send_wr wr;
  caffe_memset(sizeof(wr), 0, &wr);
  wr.wr_id = (uint64_t) this;
  wr.sg_list = &list;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = id_;
  if (!data) {
  // ctrl signal
    wr.imm_data += CTRL_ID_OFFSET;
  }

  wr.wr.rdma.remote_addr = (uint64_t) peer_->addr;
  wr.wr.rdma.rkey = peer_->rkey;

  struct ibv_send_wr *bad_wr;

  // lock the channel since there may be multiple threads calling write()
  boost::mutex::scoped_lock lock(channel_->mutex_);
  CHECK(!ibv_post_send(channel_->qp_, &wr, &bad_wr)) << "Failed to post send";

  // TODO poll only every N writes to improve performance
  for (;;) {
    ibv_wc wc;
    int ne = ibv_poll_cq(channel_->write_cq_, 1, &wc);
    CHECK_GE(ne, 0);
    if (ne) {
      CHECK(wc.wr_id == (uint64_t)this) << "Oops. Polled a Work Completion belongs to a different buffer";
      break;
    }
  }
}


}  // namespace caffe

#endif
