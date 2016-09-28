// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#include <boost/algorithm/string.hpp>
#include <glog/logging.h>

#include "caffe/caffe.hpp"
#ifndef CPU_ONLY
#include "caffe/parallel.hpp"
#include "util/socket_sync.hpp"
#else
#include "util/parallel_cpu.hpp"
#include "util/socket_sync_cpu.hpp"
#endif

#ifdef INFINIBAND
#include "util/rdma.hpp"
#include "util/rdma_sync.hpp"
#endif

#include "CaffeNet.hpp"
#include "caffe/util/hdf5.hpp"
#include "jni/com_yahoo_ml_jcaffe_CaffeNet.h"
#include "util/socket.hpp"


void SetCaffeMode(int solver_mode) {
    if (solver_mode == Caffe::GPU)
        Caffe::set_mode(Caffe::GPU);
    else Caffe::set_mode(Caffe::CPU);
}


template<typename Dtype>
void CaffeNet<Dtype>::aggregateValidationOutputs() {
  LOG(INFO) << "Iteration " << root_solver_->iter()
            << ", Testing net (#" << validation_net_id_ << ")";
  const shared_ptr<Net<Dtype> >& validation_net = root_solver_->test_nets()[validation_net_id_];
  if(root_solver_->param().test_compute_loss()){
    loss /= root_solver_->param().test_iter(validation_net_id_);
    LOG(INFO) << "Test loss: " << loss;
  }

  for (int i = 0; i < validation_score.size(); ++i) {
    const int output_blob_index =
      validation_net->output_blob_indices()[validation_score_output_id[i]];
    const string& output_name = validation_net->blob_names()[output_blob_index];
    const Dtype loss_weight = validation_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
    const Dtype mean_score = validation_score[i] / root_solver_->param().test_iter(validation_net_id_);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
              << mean_score << loss_msg_stream.str();
    
  }
  validation_score_output_id.clear();
  validation_score.clear();
  loss = 0;
}

template<typename Dtype>
void CaffeNet<Dtype>::validation(vector< Blob<Dtype>* >& input_validation_data) {
   
  input_adapter_validation_->feed(input_validation_data);
  CHECK(Caffe::root_solver());
  CHECK_NOTNULL(root_solver_->test_nets()[validation_net_id_].get())->
    ShareTrainedLayersWith(root_solver_->net().get());
  Dtype iter_loss;
  const shared_ptr<Net<Dtype> >& validation_net = root_solver_->test_nets()[validation_net_id_];
  const vector<Blob<Dtype>*>& result =
    validation_net->Forward(&iter_loss);
  if(root_solver_->param().test_compute_loss()){
    loss += iter_loss;
  }

  if (validation_score.empty()) {
    for (int j = 0; j < result.size(); ++j) {
      const Dtype* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k) {
        validation_score.push_back(result_vec[k]);
        validation_score_output_id.push_back(j);
      }
    }
  } else {
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const Dtype* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k) {
        validation_score[idx++] += result_vec[k];
      }
    }
  }
  return;
}



template<typename Dtype>
CaffeNet<Dtype>::CaffeNet(const string& solver_conf_file, const string& model_file,
        const string& state_file, int num_local_devices, int cluster_size,
        int node_rank, bool isTraining,
        int start_device_id, int validation_net_id)
    :
    solver_conf_file_(solver_conf_file),
    model_file_(model_file),
    state_file_(state_file),
    num_local_devices_(num_local_devices),
    cluster_size_(cluster_size),
    node_rank_(node_rank),
    start_device_id_(start_device_id),
    isTraining_(isTraining),
    validation_net_id_(validation_net_id) {

    validation_score_output_id.clear();
    validation_score.clear();

    num_total_devices_ = cluster_size_ * num_local_devices_;
    //read in solver parameter
    ReadSolverParamsFromTextFileOrDie(solver_conf_file, &solver_param_);
    solver_mode_ = (int) solver_param_.solver_mode();

    CHECK_GE(num_local_devices_, 1) << "number of local Devices must be greater than or equal to 1";
    CHECK_GE(cluster_size_, 0) << "cluster size must be positive";

    // grab GPU if needed
    nets_.resize(num_local_devices_);
    input_adapter_.resize(num_local_devices_);
    local_devices_.resize(num_local_devices_);
    int d = start_device_id_;
    input_adapter_validation_.reset();
    for (int i = 0; i < num_local_devices_; i++ ) {
        input_adapter_[i].reset();
        if (solver_mode_ != Caffe::GPU)
            d++;
        else {
            d = Caffe::FindDevice(d + 1);
            CHECK_GE(d, 0) << "cannot grab GPU device";
        }
        local_devices_[i] = d;
    }

    SolverParameter local_solver_param(solver_param_);
    if (local_solver_param.has_device_id()
        && (local_solver_param.device_id() != local_devices_[0])){
        LOG(WARNING) << "device " << local_solver_param.device_id() << " in the solver param not available";
    }
    local_solver_param.set_device_id(local_devices_[0]);
    LOG(INFO) << "set root solver device id to " << local_devices_[0];

    if (solver_mode_ == Caffe::GPU)
        Caffe::SetDevice(local_devices_[0]);
    SetCaffeMode(solver_mode_);
    // set number of local solvers
    // this needs to be set per thread
    // we are using num_local_devices_ here since
    // data reader needs to be initialized to this value.
    // we will switch it to num_total_devices in PreaperSolver()
    // to ensure correct gradient scaling.
    Caffe::set_solver_count(num_local_devices_);

    // turn off snapshot
    int max_iter = local_solver_param.max_iter() + 1;
    CHECK_GT(max_iter, 0);
    local_solver_param.set_snapshot(max_iter);

    test_interval = local_solver_param.test_interval();
    
    local_solver_param.set_test_interval(max_iter);
    local_solver_param.set_test_initialization(false);

    LOG(INFO) << "local_solver_param:" << local_solver_param.net();

    NetParameter net_param;
    ReadNetParamsFromTextFileOrDie(local_solver_param.net(), &net_param);
    // change the batch size for memory layer.
    // maybe need to extend to other layers.
    for (int i = 0; i < net_param.layer_size(); i++) {
        LayerParameter* layer_param = net_param.mutable_layer(i);
        // check if it has memory layer
        if (layer_param->has_memory_data_param()){
            // memory layers are not shared
            layer_param->mutable_memory_data_param()->set_share_in_parallel(false);

            //disable transform for mem data layer
            layer_param->clear_transform_param();
        }
    }
    // clean the net file so that solver will not read it anymore.
    local_solver_param.clear_net();
    // solver reads from net_param instead.
    local_solver_param.mutable_net_param()->MergeFrom(net_param);

    root_solver_.reset(caffe::SolverRegistry<Dtype>::CreateSolver(local_solver_param));
    // restore snapshot if available
    if (!state_file_.empty()) {
        if (!model_file_.empty()) {
            setLearnedNet(state_file_, model_file_);
            root_solver_->Restore(state_file_.c_str());
        }
    } else if (!model_file_.empty()) {
        copyLayers(model_file_);
    }

    //syncs_
    if (solver_mode_ == Caffe::GPU)
        syncs_.resize(num_local_devices);
    else {
      if (cluster_size > 1) {
        syncs_.resize(1);
      }
      else{
        syncs_.resize(0);
      }
      CHECK_EQ(num_local_devices, 1) << "CPU mode only allow single device";
    }
}

template<typename Dtype>
LocalCaffeNet<Dtype>::LocalCaffeNet(const string& solver_conf_file, const string& model_file,
                                    const string& state_file, int num_local_devices, bool isTraining, int start_device_id, int validation_net_id)
    : CaffeNet<Dtype>(solver_conf_file, model_file, state_file, num_local_devices, 1, 0,
                      isTraining, start_device_id, validation_net_id) {
}

#ifdef INFINIBAND
template<typename Dtype>
RDMACaffeNet<Dtype>::RDMACaffeNet(const string& solver_conf_file, const string& model_file,
          const string& state_file, int num_local_devices,
                                  int cluster_size, int node_rank, bool isTraining, int start_device_id, int validation_net_id)
    : CaffeNet<Dtype>(solver_conf_file, model_file, state_file, num_local_devices,
                      cluster_size, node_rank,isTraining, start_device_id, validation_net_id) {

    rdma_channels_.resize(this->cluster_size_);

    CHECK_EQ(this->solver_mode_, Caffe::GPU) << "RDMA connection is supported for GPU only";
    rdma_adapter_.reset(new RDMAAdapter());
    LOG(INFO)<< "RDMA adapter: " << rdma_adapter_->name();

    // The node creates a RDMA address for each node in the cluster except itself.
    // The RDMA addresses are ordered according to the rank of the peers.
    // Create channel for each peer
    for (int i = 0; i < this->cluster_size_; i++) {
        if (i != this->node_rank_)
            rdma_channels_[i].reset(new RDMAChannel(*rdma_adapter_));
    }
}
#endif

template<typename Dtype>
SocketCaffeNet<Dtype>::SocketCaffeNet(const string& solver_conf_file, const string& model_file,
            const string& state_file, int num_local_devices,
                                      int cluster_size, int node_rank, bool isTraining, int start_device_id, int validation_net_id)
    : CaffeNet<Dtype>(solver_conf_file, model_file, state_file, num_local_devices,
                      cluster_size, node_rank,isTraining, start_device_id, validation_net_id) {

    sockt_channels_.resize(this->cluster_size_);

    sockt_adapter_.reset(new SocketAdapter(&sockt_channels_));
    LOG(INFO)<< "Socket adapter: " << sockt_adapter_->address();

    // The node creates a Socket Channel for each node in the cluster except
    // itself.
    // The Socket Channels are ordered according to the rank of the peers.
    // Create channel for each peer
    for (int i = 0; i < this->cluster_size_; i++) {
        if (i != this->node_rank_)
            sockt_channels_[i].reset(new SocketChannel());
    }
}

template<typename Dtype>
CaffeNet<Dtype>::~CaffeNet() {
    int i ;

    for (i=0; i<syncs_.size(); i++)
        syncs_[i].reset();

    for (i=0; i<num_local_devices_; i++) {
        nets_[i].reset();
        input_adapter_[i].reset();
    }
    input_adapter_validation_.reset();
}

#ifdef INFINIBAND
template<typename Dtype>
RDMACaffeNet<Dtype>::~RDMACaffeNet() {
    for (int i=0; i<this->cluster_size_; i++)
        rdma_channels_[i].reset();

    rdma_adapter_.reset();
}
#endif

template<typename Dtype>
SocketCaffeNet<Dtype>::~SocketCaffeNet() {
    for (int i=0; i<CaffeNet<Dtype>::cluster_size_; i++)
        sockt_channels_[i].reset();

    sockt_adapter_.reset();
}

template<typename Dtype>
bool CaffeNet<Dtype>::isTestPhase(LayerParameter* layer_param){
    bool isTest = false;
    for (int i = 0; i < layer_param->include_size(); i++) {
        if (layer_param->include(i).phase() == TEST) {
            isTest = true;
            break;
        }
    }
    return isTest;
}

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
template <typename Dtype>
void CaffeNet<Dtype>::copyLayers(const std::string& model_list) {
    std::vector<std::string> model_names;
    boost::split(model_names, model_list, boost::is_any_of(",") );
    for (int i = 0; i < model_names.size(); ++i) {
        LOG(INFO) << "Finetuning from " << model_names[i];
        root_solver_->net()->CopyTrainedLayersFrom(model_names[i]);
        for (int j = 0; j < root_solver_->test_nets().size(); ++j) {
            root_solver_->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
        }
    }
}

template <typename Dtype>
void CaffeNet<Dtype>::setLearnedNet(const std::string& state_filename,
				    const std::string& model_filename) {
    if (state_filename.size() >= 3 &&
        state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
        setLearnedNetHDF5(state_filename, model_filename);
    } else {
        setLearnedNetBinaryProto(state_filename, model_filename);
    }
}

template <typename Dtype>
void CaffeNet<Dtype>::setLearnedNetHDF5(const std::string& state_filename,
					const std::string& model_filename) {
    hid_t file_hid = H5Fopen(state_filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    CHECK_GE(file_hid, 0) << "Couldn't open solver state file " << state_filename;
    if (H5LTfind_dataset(file_hid, "learned_net")) {
        herr_t status = H5Ldelete(file_hid, "learned_net", H5P_DEFAULT);
        CHECK_GE(status, 0)
            << "Failed to delete string dataset learned_net";
    }
    hdf5_save_string(file_hid, "learned_net", model_filename);
    H5Fclose(file_hid);
}

template <typename Dtype>
void CaffeNet<Dtype>::setLearnedNetBinaryProto(const std::string& state_filename,
            const std::string& model_filename) {
    SolverState state;
    ReadProtoFromBinaryFile(state_filename, &state);
    state.set_learned_net(model_filename);
    WriteProtoToBinaryFile(state, state_filename.c_str());
}


/**
 * retrieve the server address in which we will accept messages from peers in the cluster
 *
 * @return a collection of server addresses
 */
template<typename Dtype>
void LocalCaffeNet<Dtype>::localAddresses(vector<string>& vec) {
    vec.resize(0);
}

#ifdef INFINIBAND
template<typename Dtype>
void RDMACaffeNet<Dtype>::localAddresses(vector<string>& vec) {
    vec.resize(this->cluster_size_);
    for (int i = 0; i < this->cluster_size_; i++) {
        if (i != this->node_rank_) {
            vec[i] = rdma_channels_[i]->address();
        }
        else
            vec[i] = "";
        LOG(INFO) << i << "-th RDMA addr: " << vec[i].c_str();
    }
}
#endif

template<typename Dtype>
void SocketCaffeNet<Dtype>::localAddresses(vector<string>& vec) {
    vec.resize(this->cluster_size_);
    for (int i = 0; i < this->cluster_size_; i++) {
        if (i != this->node_rank_) {
            vec[i] = sockt_adapter_->address();
        }
        else
            vec[i] = "";
        LOG(INFO) << i << "-th Socket addr: " << vec[i].c_str();
    }
}

/**
 * establish connection among solvers and cluster peers
 *
 * @param addresses Array of addresses, whose index represents rank
 * @return true if connected successfully
 */
template<typename Dtype>
bool LocalCaffeNet<Dtype>::connect(vector<const char*>& addresses) {
#ifndef CPU_ONLY
    //When syncs_.size() == 0, we will use root_solver_ only
    if (this->syncs_.size() > 0) {
        this->syncs_[0].reset(new P2PSync<Dtype>(this->root_solver_,
                                                            NULL, this->root_solver_->param()));
        // Pair devices for map-reduce synchronization
        this->syncs_[0]->Prepare(this->local_devices_,
                                &this->syncs_);
    }
#else
    if (this->syncs_.size() > 0) {
        this->syncs_[0].reset(new P2PSyncCPU<Dtype>(this->root_solver_,
                                                            NULL, this->root_solver_->param()));
    }
#endif
    return true;
}

#ifdef INFINIBAND
template<typename Dtype>
bool RDMACaffeNet<Dtype>::connect(vector<const char*>& peer_addresses) {
    //establish RDMA connections
    for (int i = 0; i < this->cluster_size_; i++)
        if (i != this->node_rank_) {
            const char* addr = peer_addresses[i];
            string addr_str(addr, strlen(addr));
            rdma_channels_[i]->Connect(addr_str);
        }

    //set up syncs[0 ... (local_devices_-1)]
    this->syncs_[0].reset(new RDMASync<Dtype>(this->root_solver_,
                                              rdma_channels_,
                                              this->node_rank_));
    // Pair devices for map-reduce synchronization
    this->syncs_[0]->Prepare(this->local_devices_,
                            &this->syncs_);

    return true;
}
#endif

template<typename Dtype>
bool SocketCaffeNet<Dtype>::connect(vector<const char*>& peer_addresses) {
    //establish RDMA connections
    for (int i = 0; i < this->cluster_size_; i++)
        if (i != this->node_rank_) {
            const char* addr = peer_addresses[i];
            string addr_str(addr, strlen(addr));
            if(!sockt_channels_[i]->Connect(addr_str))
              return false;
        }

#ifndef CPU_ONLY
    //set up syncs[0 ... (local_devices_-1)]
    this->syncs_[0].reset(new SocketSync<Dtype>(this->root_solver_,
                                                sockt_channels_,
                                                this->node_rank_));
    // Pair devices for map-reduce synchronization
    this->syncs_[0]->Prepare(this->local_devices_,
                             &this->syncs_);
#else
    this->syncs_[0].reset(new SocketSyncCPU<Dtype>(this->root_solver_,
						   sockt_channels_,
						   this->node_rank_));
#endif
    return true;
}

/*
 * Class:     com_yahoo_ml_jcaffe_CaffeNet
 * Method:    sync
 * Signature: ()Z
 */
#ifdef INFINIBAND
template<typename Dtype>
void RDMACaffeNet<Dtype>::sync()  {
    if (this->cluster_size_ > 1)
        boost::static_pointer_cast<RDMASync<Dtype> >(this->syncs_[0])->sync(false);
}
#endif


template<typename Dtype>
void SocketCaffeNet<Dtype>::sync()  {
    if (this->cluster_size_ > 1)
#ifndef CPU_ONLY
        boost::static_pointer_cast<SocketSync<Dtype> >(this->syncs_[0])->sync(false);
#else
        boost::static_pointer_cast<SocketSyncCPU<Dtype> >(this->syncs_[0])->sync(false);
#endif
}

/**
 * retreve the device assigned to a given solver
 *
 * @param solver_index the index of a solver
 * @return device ID assiged to that solver
 */
template<typename Dtype>
int CaffeNet<Dtype>::deviceID(int solver_index) {
    if (syncs_.size() == 0)
        return root_solver_->param().device_id();
    else {
        CHECK(syncs_[solver_index]);
        return syncs_[solver_index]->solver()->param().device_id();
    }
}

/**
 * number of iterations performed previously
 *
 * @param solver_index index of our solver
 * @return initial number of iteration
 */
template<typename Dtype>
int CaffeNet<Dtype>::getInitIter(int solver_index) {
    if (syncs_.size() == 0)
        return root_solver_->iter();
    else {
        CHECK(syncs_[solver_index]);
	if (solver_index == 0)
	    return root_solver_->iter();
	else
            return syncs_[solver_index]->initial_iter();
    }
}

/**
 * max number of iterations to be performed
 *
 * @param solver_index index of our solver
 * @return max number of iteration
 */
template<typename Dtype>
int CaffeNet<Dtype>::getMaxIter(int solver_index) {
    if (syncs_.size() == 0)
        return root_solver_->param().max_iter();
    else {
        CHECK(syncs_[solver_index]);
        return syncs_[solver_index]->solver()->param().max_iter();
    }
}

/**
 * test iterations to be performed
 *
 * @param solver_index index of our solver
 * @return test iteration
 */
template<typename Dtype>
int CaffeNet<Dtype>::getTestIter(int solver_index) {
    if (syncs_.size() == 0)
      return  root_solver_->param().test_iter(0);
    else {
        CHECK(syncs_[solver_index]);
        return syncs_[solver_index]->solver()->param().test_iter(0);
    }
}

/**
 * test interval
 *
 * @param solver_index index of our solver
 * @return test interval
 */
template<typename Dtype>
int CaffeNet<Dtype>::getTestInterval() {
  return test_interval;
}

/**
 * prepare the current thread to work with a specified solver
 *
 * this function prepares solver per thread.
 * it has to be run on all threads individually.
 * @param solver_index     index of our solver
 * @return true if connected successfully
 */
template<typename Dtype>
bool CaffeNet<Dtype>::init(int solver_index, bool enableNN) {
    shared_ptr<Solver<Dtype> > solver;
    if (solver_mode_ == Caffe::CPU) {
        CHECK_EQ(solver_index, 0) << "solver_index must be 0 for local CaffeNet in CPU mode";
        solver = root_solver_;
    } else {
        CHECK(syncs_[solver_index]) << "solver was not initialized";
        solver = syncs_[solver_index]->solver();
    }
    CHECK(solver) << "solver is NULL";

    if (solver_mode_ == Caffe::GPU) {
        Caffe::SetDevice(solver->param().device_id());
    }
    SetCaffeMode(solver_mode_);

    if (enableNN) {
        CHECK(Caffe::root_solver());
        if (solver_index != 0) // all the solvers are slaves except the first one.
            Caffe::set_root_solver(false);
        // See if there is a defined seed and reset random state if so
        if (solver->param().random_seed() >= 0) {
            // Fetch random seed and modulate by device ID to make sure
            // everyone doesn't have the same seed.  We seem to have some
            // solver instability if we have everyone with the same seed
            Caffe::set_random_seed(solver->param().random_seed() + solver->param().device_id());
        }

        // data reader if exists, should already be
        // initialized with num_local_devices.
        // switch solver count to num_total_devices for correct
        // gradient scaling.
        Caffe::set_solver_count(num_total_devices_);
        // Check the mode request
        interleaved = true;
        if ( getTestInterval() == 0 && getTestIter(0) == 0) {
          interleaved = false;
        }
          
        if (isTraining_ && interleaved && (solver_index == 0)) {
          LOG(INFO) << "Interleaved";
          nets_[solver_index] = solver->test_nets()[0];
          setInputAdapter(0, nets_[solver_index]->layers()[0], true);  
          nets_[solver_index] = solver->net();
        }
        else if (isTraining_) {
          LOG(INFO) << "Training only";
          nets_[solver_index] = solver->net();
        }
        else {
          LOG(INFO) << "Test only";
          nets_[solver_index] = solver->test_nets()[0];
        }
        
        setInputAdapter(solver_index, nets_[solver_index]->layers()[0], false);
        

        CHECK(input_adapter_[solver_index].get());
    }

    return true;
}

template<typename Dtype>
void CaffeNet<Dtype>::setInputAdapter(int solver_index, shared_ptr<Layer<Dtype> > layer, bool isValidation) {
    InputAdapter<Dtype>* adapter = InputAdapterRegistry<Dtype>::MakeAdapter(layer, solver_mode_);
    CHECK(adapter != NULL);
    if (!isValidation)
      input_adapter_[solver_index].reset(adapter);
    else
      input_adapter_validation_.reset(adapter);

}

/**
 * Apply the given input data (as a array of blobs) onto the current network via the specified input blobs,
 * perform forward() and extract the output values associated with the output blob
 *
 * @param solver_index index of our solver
 * @param input_data   array of input data to be attached to input blobs
 * @param output_blobs array of output blob names
 * @return array of output data from the output blobs. null if failed
 */
template<typename Dtype>
void CaffeNet<Dtype>::predict(int solver_index,
                              vector< Blob<Dtype>* >&  input_data,
                              vector<const char*>& output_blob_names,
                              vector<Blob<Dtype>* >& output_blobs) {
    //connect input data to input adapter
    if (input_adapter_[solver_index].get()==NULL) {
        //initialize the current thread
        init(solver_index, true);
    }
    input_adapter_[solver_index]->feed(input_data);

    //invoke network's Forward operation
    CHECK(nets_[solver_index]);
    nets_[solver_index]->Forward();

    //grab the output blobs via names
    int num_features = output_blob_names.size();
    for (int i = 0; i < num_features; i++) {
        output_blobs[i] = nets_[solver_index]->blob_by_name(output_blob_names[i]).get();
    }
}

/**
 * Apply the given input data to perform 1 step of training
 *
 * @param solver_index index of our solver
 * @param input_data   array of input data to be attached to input blobs
 * @return true iff successed
 */
template<typename Dtype>
bool CaffeNet<Dtype>::train(int solver_index, vector< Blob<Dtype>* >& input_data) {
    //connect input data to input adapter
    if (input_adapter_[solver_index].get()==NULL) {
        //initialize the current thread
      init(solver_index, true);
    }

    input_adapter_[solver_index]->feed(input_data);

    //invoke network's Forward operation
    shared_ptr<Solver<Dtype> > solver;
    if (solver_mode_ == Caffe::CPU) {
        CHECK_EQ(solver_index, 0) << "solver_index must be 0 for local CaffeNet in CPU mode";
        solver = root_solver_;
    } else {
        CHECK(syncs_[solver_index]) << "solver was not initialized properly";
        solver = syncs_[solver_index]->solver();
    }

    solver->Step(1);

    return true;
}

/**
 * snapshot the model and state
 */
template<typename Dtype>
int CaffeNet<Dtype>::snapshot() {
    root_solver_->Snapshot();
    return root_solver_->iter();
}

/**
 * snapshot the model and state
 */
template<typename Dtype>
vector<string> CaffeNet<Dtype>::getValidationOutputBlobNames() {
    const shared_ptr<Net<Dtype> >& validation_net = root_solver_->test_nets()[validation_net_id_];
    int num_outputs = validation_net->num_outputs();
    const vector<int> & output_blob_indices = validation_net->output_blob_indices();
    const vector<string>& blob_names = validation_net->blob_names();
    vector<string> output_blob_names;
    for (int i = 0; i < num_outputs; i++) {
      output_blob_names.push_back(blob_names[output_blob_indices[i]]);
    }
    return output_blob_names;
}

template<typename Dtype>
vector<Blob<Dtype>*> CaffeNet<Dtype>::getValidationOutputBlobs(int length) {
    const shared_ptr<Net<Dtype> >& validation_net = root_solver_->test_nets()[validation_net_id_];
    int num_outputs = validation_net->num_outputs();
    const vector<int> & output_blob_indices = validation_net->output_blob_indices();
    const vector<string>& blob_names = validation_net->blob_names();
    vector<Blob<Dtype>*> output_blobs(length);
    for (int i = 0; i < num_outputs; i++) {
        string name = blob_names[output_blob_indices[i]];
        output_blobs[i] = validation_net->blob_by_name(name).get();
    }
    return output_blobs;
}

INSTANTIATE_CLASS(CaffeNet);
INSTANTIATE_CLASS(LocalCaffeNet);
#ifdef INFINIBAND
INSTANTIATE_CLASS(RDMACaffeNet);
#endif
INSTANTIATE_CLASS(SocketCaffeNet);
