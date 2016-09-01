// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#include <arpa/inet.h>
#include <boost/algorithm/string.hpp>
#include <netdb.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <string>
#include <vector>
#include "caffe/util/math_functions.hpp"
#include "util/socket.hpp"

namespace caffe {

void gethostnamebysinaddr(struct sockaddr_in addr,
                          char * hostname, int size) {
  getnameinfo((struct sockaddr*)&addr,
              sizeof(addr), hostname, size, NULL, 0, 0);
}

struct message_header{
  // Rank helps determine which peer sent the request
  // and is the key to locate it's local SocketChannel
  int rank;
  message_type type;  // For DIFF vs DATA
  int size;  // Payload size to follow the header
};

bool send_message_header(int sockfd, int rank, message_type mt, int ms) {
  message_header mh;
  mh.rank = rank;
  mh.type = mt;
  mh.size = ms;
  uint8_t* buffer = reinterpret_cast<uint8_t*>(&mh);
  int nsent = 0;
  int len = sizeof(mh);
  while (len > 0) {
    nsent = write(sockfd, buffer, len);
    CHECK (nsent >= 0) << "ERROR: Sending message header!";
    buffer += nsent;
    len -= nsent;
  }
  return true;
}
  
void receive_message_header(int sockfd, message_header * mh) {
  uint8_t* buffer = reinterpret_cast<uint8_t*>(mh);
  int nread = 0;
  int len = sizeof(*mh);
  while(len > 0) {
    nread = read(sockfd, buffer, len);
    CHECK (nread >= 0) << "ERROR: Reading message header!";
    buffer += nread;
    len -= nread;
  }
}
  
struct connection_details {
  int serving_fd;
  SocketAdapter* sa;
};

QueuedMessage::QueuedMessage(message_type type, int size, uint8_t* buffer) {
  this->type = type;
  this->size = size;
  this->buffer = buffer;
}

// Handle each peer connection in their own handlers
void *client_connection_handler(void *metadata) {
  struct connection_details * cd = (struct connection_details*) metadata;
  struct message_header mh;
  while (1) {
    receive_message_header(cd->serving_fd, &mh);
    // FIXME: The condition where socket server gets a connection
    // from the peer but the user didn't allocate a SocketChannel
    // for the peer.This can even happen if adapter is instantiated
    // first and SocketChannel is allocated for the peer subsequently
    // For now assume the user will maintain the right order i.e
    // allocate a SocketChannel for all peers and then instantiate
    // the adapter.
    /*      if(cd->sa->channels.size() < mh.rank || cd->sq->channels.at(mh.rank) == NULL){
            printf("ERROR:No SocketChannel assigned for the peer with rank [%d]...terminating the thread handler\n", mh.rank);
            // FIXME: Notify the client and exit
            pthread_exit();
            }
    */
    // From the peer rank locate the local SocketChannel for that
    // peer and sets it's serving_fd
    SocketChannel* sc = cd->sa->channels->at(mh.rank).get();
    sc->serving_fd = cd->serving_fd;
    uint8_t* read_buffer = new uint8_t[mh.size];
    uint8_t* marker = read_buffer;
    int cur_cnt = 0;
    int max_buff = 0;
    while (cur_cnt < mh.size) {
      if ((mh.size - cur_cnt) > 256)
        max_buff = 256;
      else
        max_buff = mh.size - cur_cnt;

      int n = read(sc->serving_fd, marker, max_buff);
      CHECK(n >= 0) << "ERROR: Reading data from client";
      marker = marker + n;
      cur_cnt = cur_cnt + n;
    }
    // Wrap the received message in an object QueuedMessage and
    // push it to the local receive queue of the peer's
    // SocketChannel
    QueuedMessage* mq = new QueuedMessage(mh.type,
                                          mh.size, read_buffer);
    if(mh.type == DIFF)
      sc->receive_queue.push(mq);
    else
      sc->receive_queue_ctrl.push(mq);
  }
  return NULL;
}

void *sockt_srvr(void *metadata) {
  SocketAdapter* sa = reinterpret_cast<SocketAdapter*>(metadata);
  // Open a server socket
  int serv_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (serv_fd < 0) {
    LOG(ERROR) << "ERROR: Could not open server socket " << INADDR_ANY;
  }

  // Initialize the socket structs
  struct sockaddr_in serv_addr;
  bzero(reinterpret_cast<char *>(&serv_addr), sizeof(serv_addr));
  // Using internet domain sockets
  serv_addr.sin_family = AF_INET;
  // Get server ip
  serv_addr.sin_addr.s_addr = INADDR_ANY;
  // Allow the server to be assigned a random free port
  serv_addr.sin_port = 0;
  char hostname[256];
  // Bind the server socket to the server host
  if (bind(serv_fd, (struct sockaddr *) &serv_addr,
           sizeof(serv_addr)) < 0) {
    gethostnamebysinaddr(serv_addr, hostname, 256);
    LOG(ERROR) << "ERROR: Unable to bind socket fd to server ["
               << hostname <<"]";
  }
  // Check what port number was assigned to the socket server
  struct sockaddr_in sin;
  socklen_t len = sizeof(sin);
  int port_no = 0;
  if (getsockname(serv_fd, (struct sockaddr *)&sin, &len) == -1) {
    LOG(ERROR) << "ERROR: getting server socket details";
  } else {
    port_no = ntohs(sin.sin_port);
    LOG(INFO) << "Assigned socket server port [" << port_no << "]";
  }
  char self_name[256];
  if (gethostname(self_name, 256) == -1)
    LOG(ERROR) << "ERROR: Could not determine self hostname ["
               << errno <<"]";

  // Set the assigned socket server port no in the socket adapter
  sa->port = port_no;
  // Listen to incoming connections on this server socket
  // (5 simultaneously)
  listen(serv_fd, 5);
  gethostnamebysinaddr(serv_addr, hostname, 256);
  LOG(INFO) << "Socket Server ready [" << hostname << "]";
  // Accept the incoming connection on the socket server
  struct sockaddr_in client_addr;
  socklen_t client_addr_len = sizeof(client_addr);
  int serving_fd;
  pthread_t thread_id;
  while ((serving_fd = accept(serv_fd, (struct sockaddr *)
                              &client_addr, &client_addr_len))) {
    char client_hostname[256];
    gethostnamebysinaddr(client_addr, client_hostname, 256);
    struct connection_details* cd = new struct connection_details;
    cd->serving_fd = serving_fd;
    cd->sa = sa;
    LOG(INFO) << "Accepted the connection from client ["
              << client_hostname << "]";
    // Create a thread per peer connection to handle
    // their communication
    if (pthread_create(&thread_id, NULL,
                       client_connection_handler,
                       reinterpret_cast<void*>(cd)) < 0) {
      LOG(ERROR) << "ERROR: Could not create thread for the "
                 << "incoming client connection ["
                 << client_hostname << "]";
      continue;
    }
  }
  CHECK (serving_fd > 0) << "ERROR: Could not accept incoming "
                         << "connection to socket server";
  // FIXME: Write threads and socket server exit/cleanup logic
  return NULL;
}

SocketAdapter::SocketAdapter(vector<shared_ptr<SocketChannel> > *
                             channels) {
  // Let SocketAdapter know the allocated SocketChannels
  // for the peers so that when the peers connect to the local
  // socket server, we can locate their SocketChannels based on
  // their ranks, received in the message header. Each vector
  // index here corresponds to the peer rank
  this->port = 0;
  this->channels = channels;
  // Start socket server
  start_sockt_srvr();
  // Wait till the socket server has
  // started with a valid port
  while (true) {
    LOG(INFO) << "Waiting for valid port ["<< this->port <<"]";
    if (this->port != 0) {
      break;
    }
    else {
      usleep(10000);
    }
  }
  LOG(INFO) << "Valid port found ["<< this->port << "]";
}

void SocketAdapter::start_sockt_srvr() {
  pthread_t thread_id;
  // Start the socket server in it's own thread as accept
  // is a blocking call
  CHECK (pthread_create(&thread_id, NULL, sockt_srvr,
                     reinterpret_cast<void*>(this)) >= 0) 
    << "ERROR: Could not start the socket server";
}

// Connect called by client with inbuilt support for retries
bool SocketChannel::Connect(string peer) {
  bool retry = true;
  int attempts = 0;
  int client_fd = 0;
  vector<string> name_port;
  boost::split(name_port, peer, boost::is_any_of(":"));
  int backoff = 1;
  while (retry && (attempts < 5)) {
    retry = false;
    if (client_fd == 0) {
      string peername = name_port.at(0).c_str();;
      string portnumber;
      if (name_port.size() > 1) 
        portnumber = name_port.at(1).c_str();
      
      LOG(INFO) << "Trying to connect with ...["
                << peername <<":"
                << portnumber << "]";
      client_fd = connect_to_peer(peername,
                                  portnumber);
      if (!client_fd) {
        retry = true;
      } else {
        // On success update the local peer's
        // SocketChannel with the connection details
        this->client_fd = client_fd;
        this->peer_name = name_port.at(0);
        this->port_no = atoi(name_port.at(1).c_str());
      }
    }
    attempts++;
    // Retry after 10 secs
    usleep(backoff*1000000);
    backoff = backoff * 2;
  }
  if (retry)
    return false;
  
  return true;
}

// Real connect call without retries (called by connect above)
int SocketChannel::connect_to_peer(string to_peer, string to_port) {
  // Parse socket server details to connect
  struct hostent* serv = gethostbyname(to_peer.c_str());
  if (serv == NULL) {
    LOG(ERROR) << "ERROR: No peer by name [" << to_peer.c_str() << "]";
    return 0;
  }
  int serv_port = atoi(to_port.c_str());

  // Create a client socket
  int client_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (client_fd < 0) {
    LOG(ERROR) << "ERROR: Unable to open client socket to peer ["
               << to_peer.c_str() << "]";
    return 0;
  }
  // Initialize the socket structs
  struct sockaddr_in serv_addr;
  bzero(reinterpret_cast<char *>(&serv_addr), sizeof(serv_addr));
  // Domain socket for internet
  serv_addr.sin_family = AF_INET;
  // Convert to network byte order
  serv_addr.sin_port = htons(serv_port);
  // Get the server ip address
  bcopy(reinterpret_cast<char *>(serv->h_addr),
        reinterpret_cast<char *>(&serv_addr.sin_addr.s_addr),
        serv->h_length);
  char server_hostname[256];
  gethostnamebysinaddr(serv_addr, server_hostname, 256);
  if (connect(client_fd, (struct sockaddr *) &serv_addr,
              sizeof(serv_addr)) < 0) {
    LOG(ERROR) << "ERROR: Unable to connect to socket server ["
               << server_hostname <<"]";
    close(client_fd);
    return 0;
  }

  LOG(INFO) << "Connected to server [" << server_hostname
            <<":"<< serv_port<< "] with client_fd ["<< client_fd << "]";
  return client_fd;
}

SocketChannel::SocketChannel() {
  this->client_fd = 0;
  this->serving_fd = 0;
  this->size = 0;
}

SocketChannel::~SocketChannel() {
  this->client_fd = 0;
  this->serving_fd = 0;
  this->peer_name.clear();
  this->size = 0;
}

SocketBuffer::SocketBuffer(int rank, SocketChannel* channel,
                           uint8_t* buffer, size_t size, uint8_t* addr) {
  this->rank = rank;
  this->channel_ = channel;
  this->buffer_ = buffer;
  this->size_ = size;
  this->addr_ = addr;
}

void SocketBuffer::Write(bool data) {
  uint8_t* marker = NULL;
  size_t size = 0;
  message_type mt = CTRL;

  if (data) {
#ifndef CPU_ONLY
    // Copy the buffer to be sent from GPU
    cudaMemcpy(this->buffer_, this->addr_, this->size_,  // NOLINT(caffe/alt_fn)
               cudaMemcpyDeviceToHost);  // NOLINT(caffe/alt_fn)
#endif
    marker = reinterpret_cast<uint8_t*>(this->buffer());
    size = this->size_;
    mt = DIFF;
  } 

  boost::mutex::scoped_lock lock(this->channel_->mutex_);
  
  CHECK (send_message_header(channel_->client_fd,
                              this->rank, mt, size)) << "ERROR: Sending message header from client";
  int cur_cnt = 0;
  int max_buff = 0;
  while (cur_cnt < size) {
    if ((size - cur_cnt) > 256)
      max_buff = 256;
    else
      max_buff = size - cur_cnt;

    int n = write(channel_->client_fd, marker, max_buff);
    CHECK(n >= 0) << "ERROR:Sending data from client";
    marker = marker + n;
    cur_cnt = cur_cnt + n;
  }
}

SocketBuffer* SocketBuffer::Read(bool data) {
  // Pop the message from local queue
  QueuedMessage* qm = NULL;
  if(data) {
    qm = reinterpret_cast<QueuedMessage*>
      (this->channel_->receive_queue.pop());
#ifndef CPU_ONLY
    // Copy the received buffer to GPU memory
    CUDA_CHECK(cudaMemcpy(this->addr(), qm->buffer,  // NOLINT(caffe/alt_fn)
               qm->size, cudaMemcpyHostToDevice));  // NOLINT(caffe/alt_fn)
#else
    //caffe_copy(qm->size, qm->buffer, this->addr_);
    memcpy(this->addr_, qm->buffer, qm->size);
#endif
  } else {
    qm = reinterpret_cast<QueuedMessage*>
      (this->channel_->receive_queue_ctrl.pop());
  }
  // Free up the buffer and the wrapper object
  if(data)
    delete qm->buffer;
  delete qm;
  return this;
}

Socket::Socket(const string& host, int port, bool listen) {
  addrinfo *res;
  addrinfo hints;
  caffe_memset(sizeof(addrinfo), 0, &hints);
  if (listen) {
    hints.ai_flags = AI_PASSIVE;
  }
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  string p = boost::lexical_cast<string>(port);
  const char* server = host.size() ? host.c_str() : NULL;
  int n = getaddrinfo(server, p.c_str(), &hints, &res);
  CHECK_GE(n, 0)<< gai_strerror(n) << " for " << host << ":" << port;
  fd_ = -1;
  for (addrinfo* t = res; t; t = t->ai_next) {
    fd_ = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
    if (fd_ >= 0) {
      if (listen) {
        int n = 1;
        setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &n, sizeof n);
        if (!bind(fd_, t->ai_addr, t->ai_addrlen))
          break;
      } else {
        if (!connect(fd_, t->ai_addr, t->ai_addrlen))
          break;
      }
      close(fd_);
      fd_ = -1;
    }
  }
  freeaddrinfo(res);
  string verb(listen ? "listen" : "connect");
  CHECK_GE(fd_, 0) << "Could not " << verb << " to " << host << ":" << port;
  if (listen) {
    LOG(INFO)<< "Listening to port " << port;
    ::listen(fd_, 1);
  }
}
  
  Socket::~Socket() {
    close(fd_);
  }

  shared_ptr<Socket> Socket::accept() {
    int fd = ::accept(fd_, NULL, 0);
    CHECK_GE(fd,  0) << "Socket accept failed";
    return shared_ptr<Socket>(new Socket(fd));
  }

  size_t Socket::read(void* buff, size_t size) {
    return ::read(fd_, buff, size);
  }

  size_t Socket::write(void* buff, size_t size) {
    return ::write(fd_, buff, size);
  }

}  // namespace caffe

