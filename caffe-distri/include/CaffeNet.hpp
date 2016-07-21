// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#ifndef CAFFE_DISTRI_CAFFENET_HPP_
#define CAFFE_DISTRI_CAFFENET_HPP_

#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#include "util/InputAdapter.hpp"
#include "util/parallel_cpu.hpp"
#include "util/rdma.hpp"
#include "util/socket.hpp"

void SetCaffeMode(int solver_mode);

template<typename Dtype>
class CaffeNet {
  protected:
    string solver_conf_file_, model_file_, state_file_;
    int num_local_devices_, num_total_devices_;
    int cluster_size_, node_rank_, start_device_id_;
    bool isTraining_;
    bool interleaved;
    SolverParameter solver_param_;
    int solver_mode_;
    int validation_net_id_;
    vector<int> local_devices_;
    shared_ptr<Solver<Dtype> > root_solver_;
#ifndef CPU_ONLY
    vector<shared_ptr<P2PSync<Dtype> > > syncs_;
#else
    vector<shared_ptr<P2PSyncCPU<Dtype> > > syncs_;
#endif
    vector<shared_ptr<Net<Dtype> > > nets_;
    vector<shared_ptr<InputAdapter<Dtype> > > input_adapter_;
    shared_ptr<InputAdapter<Dtype> > input_adapter_validation_;
    vector<Dtype> validation_score;
    vector<int> validation_score_output_id;
    Dtype loss;
    int test_interval;

  public:
  
    /**
     * constructor of CaffeNet.
     * Solvers will be constructed, and each solver will be assigned a device
     * Devices will be assigned to each solver
     *
     * @param solver_conf_file file path for solver's configuration
     * @param model_file file path for model file
     * @param state_file file path for state file
     * @param num_local_devices     # of local devices
     * @param cluster_size     size of cluster
     * @param node_rank        this node's rank in the cluster
     * @param isTraining       true for training, false otherwise
     * @param start_device_id  the start ID of device. default: -1
     * @param validation_net_id  validation net id. default: 0
     */
    CaffeNet(const string& solver_conf_file,
             const string& model_file,
             const string& state_file,
             int num_local_devices,
             int cluster_size,
             int node_rank,
             bool isTraining,
             int start_device_id,
             int validation_net_id);

    /**
       destructor
    */
    virtual ~CaffeNet();

    /**
     * retrieve the server address in which we will accept messages from peers in the cluster
     *
     * @return a collection of server addresses
     */
    virtual void localAddresses(vector<string>& vec) = 0;

    /**
     * establish connection with cluster peers
     *
     * @param addresses Array of addresses, whose index represents rank
     * @return true if connected successfully
     */
    virtual bool connect(vector<const char*>& addresses) = 0; //list of addresses, whose index represents rank

    virtual void sync() { }

    /**
     * retreve the device assigned to a given solver
     *
     * @param solver_index the index of a solver
     * @return device ID assiged to that solver
     */
    virtual int deviceID(int solver_index);

    /**
     * prepare the current thread to work with a specified solver
     *
     * @param solver_index     index of our solver
     * @param enableNN         flag indicate whether neural network should be set up or not
     * @return true if connected successfully
     */
    virtual bool init(int solver_index, bool enableNN);

    /**
     * Apply the given input data (as a array of blobs) onto the current network via the specified input blobs,
     * perform forward() and extract the output values associated with the output blob
     *
     * @param solver_index index of our solver
     * @param input_data   array of input data to be attached to input blobs
     * @param output_blobs array of output blob names
     * @return array of output data from the output blobs. null if failed
     */
    virtual void predict(int solver_index, vector< Blob<Dtype>* >&  input_data,
        vector<const char*>& output_blob_names, vector<Blob<Dtype>* >& output_blobs);

    /**
     * Apply the given input data to perform 1 step of training
     *
     * @param solver_index index of our solver
     * @param input_data   array of input data to be attached to input blobs
     * @return true if success
     */
    virtual bool train(int solver_index, vector< Blob<Dtype>* >& input_data);

    /**
     * Apply the given input data to perform test_iter number of interleaved validation
     *
     * @param input_data   array of validation input data to be attached to input blobs
     */
  virtual void validation(vector< Blob<Dtype>* >& input_data);

  /** Calculate the aggregate loss for all the validation iterations
   */
  virtual void aggregateValidationOutputs();
    /**
     * number of iterations performed previously
     *
     * @param solver_index index of our solver
     * @return initial number of iteration
     */
    virtual int getInitIter(int solver_index);

    /**
     * max number of iterations to be performed
     *
     * @param solver_index index of our solver
     * @return max number of iteration
     */
    virtual int getMaxIter(int solver_index);

    /**
     * snapshot the model and state
     */
    virtual int snapshot();

    /**
     * get the validation net output blob names
     */
    vector<string> getValidationOutputBlobNames();


    /**
     * get the validation net output blobs
     * @param length no. of output blobs
     */
    vector<Blob<Dtype>*> getValidationOutputBlobs(int length);

    /**
     * get the no. of test iterations
     */
    virtual int getTestIter(int solver_index);

    /**
     * get the test interval
     */
    virtual int getTestInterval();

  protected:
    /**
       Store the previously learned network into a given file
       @param state_filename state file that contains previously learned state
       @param model_filename model file into which we will save the learned
       network
       @param is this for setting train or interleave/validation
    */
    virtual void setLearnedNet(const std::string& state_filename, const std::string& model_filename);
  void setInputAdapter(int solver_index, shared_ptr<Layer<Dtype> > layer, bool isValidation);

  private:
    void setLearnedNetHDF5(const std::string& state_filename, const std::string& model_filename);
    void setLearnedNetBinaryProto(const std::string& state_filename, const std::string& model_filename);
    bool isTestPhase(LayerParameter* layer_param);
    void copyLayers(const std::string& model_list);
};

template<typename Dtype>
class LocalCaffeNet : public CaffeNet<Dtype> {
  public:
    /**
     * constructor of LocalCaffeNet.
     * Solvers will be constructed, and each solver will be assigned a device
     * Devices will be assigned to each solver
     *
     * @param solver_conf_file file path for solver's configuration
     * @param model_file file path for model file
     * @param state_file file path for state file
     * @param num_local_devices     # of local devices
     * @param isTraining       true for training, false otherwise
     * @param start_device_id  the start ID of device. default: -1
     */
    LocalCaffeNet(const string& solver_conf_file,
                  const string& model_file,
                  const string& state_file,
                  int num_local_devices,
                  bool isTraining,
                  int start_device_id, 
                  int validation_net_id);

    /**
       destructor
    */
    virtual ~LocalCaffeNet() {};

    /**
     * retrieve the server address in which we will accept messages from peers in the cluster
     *
     * @return a collection of server addresses
     */
    virtual void localAddresses(vector<string>& vec);

    /**
     * establish connection with cluster peers
     *
     * @param addresses Array of addresses, whose index represents rank
     * @return true if connected successfully
     */
    virtual bool connect(vector<const char*>& addresses); //list of addresses, whose index represents rank
};

#ifdef INFINIBAND
template<typename Dtype>
class RDMACaffeNet : public CaffeNet<Dtype> {
  protected:
    shared_ptr<RDMAAdapter> rdma_adapter_;
    vector<shared_ptr<RDMAChannel> > rdma_channels_;
  public:
    /**
     * constructor of LocalCaffeNet.
     * Solvers will be constructed, and each solver will be assigned a device
     * Devices will be assigned to each solver
     *
     * @param solver_conf_file file path for solver's configuration
     * @param model_file file path for model file
     * @param state_file file path for state file
     * @param num_local_devices     # of local devices
     * @param node_rank        this node's rank in the cluster
     * @param isTraining       true for training, false otherwise
     * @param start_device_id  the start ID of device. default: -1
     */
    RDMACaffeNet(const string& solver_conf_file,
                 const string& model_file,
                 const string& state_file,
                 int num_local_devices,
                 int cluster_size,
                 int node_rank,
                 bool isTraining,
                 int start_device_id,
                 int validation_net_id);

    /**
       destructor
    */
    virtual ~RDMACaffeNet();

    /**
     * retrieve the server address in which we will accept messages from peers in the cluster
     *
     * @return a collection of server addresses
     */
    virtual void localAddresses(vector<string>& vec);

    /**
     * establish connection with cluster peers
     *
     * @param addresses Array of addresses, whose index represents rank
     * @return true if connected successfully
     */
    virtual bool connect(vector<const char*>& addresses); //list of addresses, whose index represents rank

    virtual void sync();
};
#endif

template<typename Dtype>
class SocketCaffeNet : public CaffeNet<Dtype> {
 protected:
  shared_ptr<SocketAdapter> sockt_adapter_;
  vector<shared_ptr<SocketChannel> > sockt_channels_;
 public:
  /**
   * constructor of SocketCaffeNet.
   * Solvers will be constructed, and each solver will be assigned a device
   * Devices will be assigned to each solver
   *
   * @param solver_conf_file file path for solver's configuration
   * @param model_file file path for model file
   * @param state_file file path for state file
   * @param num_local_devices     # of local devices
   * @param node_rank        this node's rank in the cluster
   * @param isTraining       true for training, false otherwise
   * @param start_device_id  the start ID of device. default: -1
   */
  SocketCaffeNet(const string& solver_conf_file,
	       const string& model_file,
	       const string& state_file,
	       int num_local_devices,
	       int cluster_size,
	       int node_rank,
	       bool isTraining,
               int start_device_id,
               int validation_net_id);

  /**
       destructor
  */
  virtual ~SocketCaffeNet();

  /**
   * retrieve the server address in which we will accept messages from peers in the cluster
   *
   * @return a collection of server addresses
   */
  virtual void localAddresses(vector<string>& vec);

  /**
   * establish connection with cluster peers
   *
   * @param addresses Array of addresses, whose index represents rank
   * @return true if connected successfully
   */
  virtual bool connect(vector<const char*>& addresses); //list of addresses, whose index represents rank

  virtual void sync();
};

#endif
