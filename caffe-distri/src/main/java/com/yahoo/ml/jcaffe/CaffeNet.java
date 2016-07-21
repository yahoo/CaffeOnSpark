// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.jcaffe;

import java.io.IOException;

import caffe.Caffe.*;

/**
 * CaffeNet is the primary class for JVM layer to interact with Caffe (C++).
 * The basic usage:
 * (1) Construct a CaffeNet object with 1+ solvers
 * (2) Invoke CaffeNet.localAddresses() to get listening addresses of this CaffeNet
 * (3) Invoke CaffeNet.connect(remote_addresses) to establish connection among solvers and peers
 * (4) Invoke CaffeNet.train() or CaffeNet.predict() to perform training or test
 * (5) Invoke CaffeNet.snapshot() periodically to save training state into file systems,
 *     with file names defined as CaffeNet.snapshotFilename().
 */
public class CaffeNet extends BaseObject {
    public static final int NONE = 0;
    public static final int RDMA = 1;
    public static final int SOCKET = 2;
    private final SolverParameter solverParameter_;

    /**
     * constructor of CaffeNet.
     *
     * Solvers are constructed, and each solver will be assigned a device
     * Devices will be assigned to each solver.
     *
     * @param solver_conf_file file path for solver's configuration
     * @param input_model_file file path for model file
     * @param input_state_file file path for state file
     * @param num_local_devices     # of local devices
     * @param cluster_size     size of cluster
     * @param node_rank           my rank in the cluster
     * @param isTraining       true for training, false otherwise
     * @param connection_type  connection type among the servers
     * @param start_device_id  the start ID of device. default: -1
     * @param validation_net_id validation net id. default: 0
     */
    public CaffeNet(String solver_conf_file,
                    String input_model_file,
                    String input_state_file,
                    int num_local_devices,
                    int cluster_size,
                    int node_rank,
                    boolean isTraining,
                    int connection_type,
                    int start_device_id,
		    int validation_net_id) throws IOException {
        solverParameter_ = Utils.GetSolverParam(solver_conf_file);
        if (!allocate(solver_conf_file, input_model_file, input_state_file,
                num_local_devices, cluster_size, node_rank, isTraining,
		      connection_type, start_device_id, validation_net_id))
            throw new RuntimeException("Failed to create CaffeNet object");
    }

    private native boolean allocate(String solver_conf_file,
                                    String input_model_file,
                                    String input_state_file,
                                    int num_local_devices,
                                    int cluster_size,
                                    int node_rank,
                                    boolean isTraining,
                                    int connection_type,
                                    int start_device_id,
				    int validation_net_id);

    @Override
    protected native void deallocate(long address);

    /**
     * establish connection among solvers and cluster peers
     *
     * @param addresses Array of addresses, whose index represents rank
     * @return true if connected successfully
     */
    public native boolean connect(String[] addresses); //list of addresses, whose index represents rank

    /**
     * Alignment with all cluster members
     * @return true if successfully
     */
    public native boolean sync();

    /**
     * initialize the current thread to work with a specified solver.
     *
     * This ensure the current thread will be assigned with the right CPU/GPU mode, and the assigned device.
     * If enableNN==true, we will also set up neural network connection with input adapter of data layers.
     * 
     * For training/test threads, this method will be invoked implicitly.
     * You should invoke init(solver_index, false) in transformer threads.
     * @param solver_index     index of our solver
     * @param enableNN         should neural network be set up for training/test?
     * @return true if successful
     */
    public native boolean init(int solver_index, boolean enableNN);

    /* conveninent method for transformer initialization */
    public boolean init(int solver_index)  {
        return init(solver_index, false);
    }

    /**
     * Apply the given input data (as a array of blobs) onto the current network via the specified input blobs,
     * perform forward() and extract the output values associated with the output blob
     *
     * If this thread has not been initd, we will invoke init(solver_index, true).
     * @param solver_index index of our solver
     * @param data   array of input data to be attached to input blobs
     * @param output_blobnames array of output blob names
     * @return output data from the output blobs. null if failed
     */
    public native FloatBlob[] predict(int solver_index, FloatBlob[] data, String[] output_blobnames);

    /**
     * Apply the given input data to perform 1 step of training
     *
     * If this thread has not been initialize, we will invoke init(solver_index, true).
     * @param solver_index index of our solver
     * @param data   array of input data to be attached to input blobs
     * @return true iff successed
     */
    public native boolean train(int solver_index, FloatBlob[] data);

    /**
     * retrieve the server address in which we will accept messages from peers in the cluster
     *
     * @return the server address
     */
    public native String[] localAddresses();

    /**
     * retreve the device assigned to a given solver
     *
     * @param solver_index the index of a solver
     * @return device ID assiged to that solver
     */
    public native int deviceID(int solver_index);

    /**
     * number of iterations performed previously
     *
     * @param solver_index index of our solver
     * @return initial number of iteration
     */
    public native int getInitIter(int solver_index);

    /**
     * max number of iterations to be performed
     *
     * @param solver_index index of our solver
     * @return max number of iteration
     */
    public native int getMaxIter(int solver_index);


    /**
     * test iterations to be performed
     *
     * @param solver_index index of our solver
     * @return test iterations
     */
    public native int getTestIter(int solver_index);

    /**
     * test interval 
     *
     * @return test interval
     */
    public native int getTestInterval();

    /**
     * snapshot the model and state
     * @return iteration ID for which the snapshot was performed; -1 if failed
     */
    public native int snapshot();

    /**
     * get the validation net output blob names
     * @return comma separate string of output blob names.
     */
    public native String[] getValidationOutputBlobNames();


    /**
     * get the test net output blobs
     * @param length no. of output blobs
     * @return array of output blobs.
     */
    public native FloatBlob[] getValidationOutputBlobs(int length);

    /**
     * get the file name of mode or state snapshot
     * @param iter iteration ID
     * @param isState true for state, false for model
     * @return file path
     */
    public String snapshotFilename(int iter, boolean isState) {
        if (iter < 0) return null;

        StringBuilder extension;
        if (isState) {
            extension = new StringBuilder(".solverstate");
        } else {
            extension = new StringBuilder(".caffemodel");
        }
        if (solverParameter_.getSnapshotFormat() == SolverParameter.SnapshotFormat.HDF5) {
            extension.append(".h5");
        }

        return  solverParameter_.getSnapshotPrefix() + "_iter_"+ iter + extension.toString();
    }

  /**
   * Apply the given input data to perform 1 step of validation
   *
   * @param validation_data array of input validation data to be attached to input blobs
   */
  public native void validation(FloatBlob[] input_validation_data);

  /**
   * Compute the aggregate of all the validation scores and the final loss etc
   *
   * @param  index of our validation net
   */  
  public native void aggregateValidationOutputs();

};
