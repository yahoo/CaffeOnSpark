// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#include "caffe/util/blocking_queue.cpp"
#include "util/socket.hpp"

#ifdef INFINIBAND
#include "util/rdma.hpp"
#endif

namespace caffe {

template class BlockingQueue<QueuedMessage*>;

#ifdef INFINIBAND
template class BlockingQueue<ibv_mr*>;
template class BlockingQueue<RDMABuffer*>;
#endif

}  // namespace caffe
