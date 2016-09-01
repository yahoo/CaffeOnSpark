// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#ifndef CAFFE_DISTRI_SOCKET_HPP_
#define CAFFE_DISTRI_SOCKET_HPP_

#include <stdio.h>
#include <map>
#include <string>
#include <vector>
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/util/blocking_queue.hpp"
#include <boost/thread.hpp>

using std::vector;
using std::map;
using std::string;

namespace caffe {
class SocketChannel;
class SocketAdapter {
 public:
  volatile int port;
  explicit SocketAdapter(vector<shared_ptr<SocketChannel> > * channels);
  vector<shared_ptr<SocketChannel> > *channels;
  void start_sockt_srvr();
  string address(){
    char self_name[256];
    char port_buf[256];
    gethostname(self_name, 256);
    snprintf(port_buf, sizeof(port_buf), "%d", port);
    string address = self_name;
    address +=":";
    address += port_buf;
    return address;
  }
};

enum message_type {DIFF, DATA, CTRL};
class QueuedMessage {
 public:
  message_type type;
  int size;
  uint8_t* buffer;
  QueuedMessage(message_type type, int size, uint8_t* buffer);
};

class SocketBuffer;
class SocketChannel {
 private:
  int connect_to_peer(string to_peer, string to_port);
 public:
  SocketChannel();
  ~SocketChannel();
  bool Connect(string peer);
  int client_fd;
  caffe::BlockingQueue<QueuedMessage*> receive_queue;
  caffe::BlockingQueue<QueuedMessage*> receive_queue_ctrl;
  int serving_fd;
  int port_no;
  string peer_name;
  size_t size;
  mutable boost::mutex mutex_;
};

class SocketBuffer {
 public:
  SocketBuffer(int rank, SocketChannel* channel,
               uint8_t* buffer, size_t size, uint8_t* addr);
  uint8_t* addr() const {
    return addr_;
  }
  uint8_t* buffer() const {
    return buffer_;
  }
  const size_t size() const {
    return size_;
  }
  // Synchronously writes content to remote peer
  void Write(bool data=true);
  SocketBuffer* Read(bool data=true);
 protected:
  SocketChannel* channel_;
  uint8_t* addr_;
  uint8_t* buffer_;
  /*const*/ size_t size_;
  int rank;
};

class Socket {
 public:
    explicit Socket(const string &host, int port, bool listen);
    ~Socket();
    int descriptor() { return fd_; }

    shared_ptr<Socket> accept();
    size_t read(void *buff, size_t size);
    size_t write(void *buff, size_t size);

    uint64_t readInt() {
        // TODO loop for partial reads or writes
        uint64_t value;
        CHECK_EQ(read(&value, sizeof(uint64_t)), sizeof(uint64_t));
        return value;
    }
    void writeInt(uint64_t value) {
        CHECK_EQ(write(&value, sizeof(uint64_t)), sizeof(uint64_t));
    }
    string readStr() {
        size_t size = readInt();
        string str(size, ' ');
        CHECK_EQ(read(&str[0], size), size);
        return str;
    }
    void writeStr(const string &str) {
        writeInt(str.size());
        CHECK_EQ(write(const_cast<void*>(reinterpret_cast<const void *>
                      (str.c_str())), str.size()), str.size());
    }

 protected:
    explicit Socket(int fd) : fd_(fd) { }
    int fd_;

    DISABLE_COPY_AND_ASSIGN(Socket);
};
}  // namespace caffe

#endif
