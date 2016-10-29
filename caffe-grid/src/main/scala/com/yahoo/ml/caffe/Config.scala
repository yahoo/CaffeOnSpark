// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import caffe.Caffe._
import com.yahoo.ml.jcaffe.{CaffeNet, Utils}

import org.apache.commons.cli.{BasicParser, CommandLine, Options}
import org.apache.spark.SparkContext
import org.slf4j.LoggerFactory

/**
 * CaffeOnSpark configuration
 */
class Config(sc: SparkContext) extends Serializable {
  @transient private var _log = LoggerFactory.getLogger(this.getClass)
  private var _protoFile = ""
  private var _isTraining = false
  private var _isTest = false
  private var _isFeature = false
  private var _label = "label"
  private var _modelPath = ""
  private var _outputPath = ""
  private var _devices = 0
  private var _isRddPersistent = false
  private var _snapshotStateFile = ""
  private var _snapshotModelFile = ""
  private var _resize = false
  private var _connection = 0
  private var _clusterSize = 0
  private var _train_data_layer_id = -1
  private var _test_data_layer_id = -1
  private var _transform_thread_per_device = 1
  private var _features = Array[String]()
  private var _outputFormat = "json"
  private var _imageRoot = ""
  private var _labelFile = ""
  private var _lmdb_partitions = 0
  private var _imageCaptionDFDir = ""
  private var _vocabDir = ""
  private var _embeddingDFDir = ""
  private var _captionFile = ""
  private var _captionLength = 20
  private var _vocabSize = -1
  @transient private var _solverParameter: SolverParameter = null
  @transient private var _netParam: NetParameter = null

  def log = {
    if (_log == null) {
      _log = LoggerFactory.getLogger(this.getClass)
    }
    _log
  }

  /**
   * Get file location of solver protocol file
   */
  def protoFile = _protoFile

  /**
   * Set file location of solver protocol file
   */
  def protoFile_=(value: String): Unit = {
    _protoFile = value
    if (value == null || value.length() == 0) {
      log.warn("solver protofile isn't given")
    } else {
      //solver parameter
      _solverParameter = Utils.GetSolverParam(value)
      _netParam = Utils.GetNetParam(solverParameter.getNet())

      //set train_data_layer_id and test_data_layer_id
      var layerId = 0
      while (train_data_layer_id < 0 || test_data_layer_id < 0) {
        val layerParameter = _netParam.getLayer(layerId)
        for (i <- 0 until layerParameter.getIncludeCount()) {
          if (layerParameter.getInclude(i).getPhase() == Phase.TRAIN)
            _train_data_layer_id = layerId
          if (layerParameter.getInclude(i).getPhase() == Phase.TEST)
            _test_data_layer_id = layerId
        }

        layerId += 1
      }
    }
  }

  /**
   * true if we perform model training
   */
  def isTraining = _isTraining

  /**
   * Set flag indicate whether we want to perform model training or not
   */
  def isTraining_=(value: Boolean): Unit = _isTraining = value

  /**
   * true if we perform model test
   */
  def isTest = _isTest

  /**
   * Set flag indicate whether we want to perform model test or not
   */
  def isTest_=(value: Boolean): Unit = {
    // check test or feature
    // Use the one which was latest set
    if (value && isFeature) {
      log.warn("both -test and -features are found, we will do test only (as it is latest), disabling feature mode.")
      _isFeature = false
    }
    _isTest = value
  }

  /**
   * true if we perform feature extraction
   */
  def isFeature = _isFeature

  /**
   * flag indicate whether we want to perform feature extraction or not
   */
  def isFeature_=(value: Boolean): Unit = {
    // check test or feature
    // Use the one which was latest set
    if (value && isTest) {
      log.warn("both -test and -features are found, we will do features only (as it is latest), disabling test mode.")
      _isTest = false
    }
    _isFeature = value
  }

  /**
   * Get label blob name
   */
  def label = _label

  /**
   * Set label blob name
   */
  def label_=(value: String): Unit = {
    //expand features with label
    if (features != null && value != null && !value.isEmpty() && !features.contains(value)) {
      _features = features :+ value
    }
    _label = value
  }

  /**
   * Get HDFS path for model file
   */
  def modelPath = _modelPath

  /**
   * Set HDFS path for model file
   */
  def modelPath_=(value: String): Unit = {
    if (value == null || value.length() == 0) {
      log.warn("modelPath is not specified")
    }
    _modelPath = value
  }

  /**
   * Get HDFS path for output file for test results etc
   */
  def outputPath = _outputPath

  /**
   * Set HDFS path for output file for test results etc
   */
  def outputPath_=(value: String): Unit = _outputPath = value

  /**
   * Get # of GPUs/CPUs per Spark executor
   */
  def devices = _devices

  /**
   * Set # of GPUs/CPUs per Spark executor
   */
  def devices_=(value: Int): Unit = _devices = value

  /**
   * True if training RDD should be persistent
   */
  def isRddPersistent = _isRddPersistent

  /**
   * Set flag indicate whether training RDD should be persistent or not
   */
  def isRddPersistent_=(value: Boolean): Unit = _isRddPersistent = value

  /**
   * Get HDFS path for snapshot states
   */
  def snapshotStateFile = _snapshotStateFile

  /**
   * Set HDFS path for snapshot states
   */
  def snapshotStateFile_=(value: String): Unit = _snapshotStateFile = value

  /**
   * Get HDFS path for snapshot models
   */
  def snapshotModelFile = _snapshotModelFile

  /**
   * Set HDFS path for snapshot models
   */
  def snapshotModelFile_=(value: String): Unit = _snapshotModelFile = value

  /**
   * true if input image will be resized
   */
  def resize = _resize

  /**
   * Set flag for resizing input image
   */
  def resize_=(value: Boolean): Unit = _resize = value

  /**
   * Get network interface for connection among Spark executors.
   * 0 : NONE, 1: RDMA, 2: SOCKET
   */
  def connection = _connection

  /**
   * Set network interface for connection among Spark executors.
   * 0 : NONE, 1: RDMA, 2: SOCKET
   */
  def connection_=(value: Int) = _connection = value

  /**
   * Get # of executors in the cluster
   */
  def clusterSize = _clusterSize

  /**
   * Set # of executors in the cluster
   */
  def clusterSize_=(value: Int) = _clusterSize = value

  /**
   * Get layer ID of training data source
   */
  def train_data_layer_id = _train_data_layer_id

  /**
   * Set layer ID of training data source
   */
  def test_data_layer_id = _test_data_layer_id

  /**
   * Get # of transformer threads per device
   */
  def transform_thread_per_device = _transform_thread_per_device

  /**
   * Set # of transformer threads per device
   */
  def transform_thread_per_device_=(value: Int) = _transform_thread_per_device = value

  /**
   * Get blob names of feature output blobs
   */
  def features = _features

  /**
   * Set blob names of feature output blobs
   */
  def features_=(value: Array[String]) =
    if (value != null && label != null && !label.isEmpty() && !value.contains(label)) {
      //expand features with label
      _features = value :+ label
    } else {
      _features = value
    }

  /**
   * Get output dataframe format.
   */
  def outputFormat = _outputFormat

  /**
   * Set output dataframe format.
   */
  def outputFormat_=(value: String) = _outputFormat = value

  /**
   * Get the path of input image folder
   */
  def imageRoot = _imageRoot

  /**
   * Set the path of input image folder
   */
  def imageRoot_=(value: String) = _imageRoot = value

  /**
   * Get the input label file
   */
  def labelFile = _labelFile

  /**
   * Set the input label file
   */
  def labelFile_=(value: String) = _labelFile = value

  /**
   * Get number of LMDB partitions
   */
  def lmdb_partitions = _lmdb_partitions

  /**
   * Set number of LMDB partitions
   */
  def lmdb_partitions_=(value: Int) = _lmdb_partitions = value

  /**
    * Get the input caption file
    */
  def captionFile = _captionFile

  /**
    * Set the input caption file
    */
  def captionFile_=(value: String) = _captionFile = value

  /**
    * Get the input caption file
    */
  def captionLength = _captionLength

  /**
    * Set the input caption file
    */
  def captionLength_=(value: Int) = _captionLength = value

  /**
    * Get the input caption file
    */
  def vocabSize = _vocabSize

  /**
    * Set the input caption file
    */
  def vocabSize_=(value: Int) = _vocabSize = value

  /**
    * Get image caption dataframe directory
    */
  def imageCaptionDFDir = _imageCaptionDFDir

  /**
    * Set image caption dataframe directory
    */
  def imageCaptionDFDir_=(value: String) = _imageCaptionDFDir = value


  /**
    * Get vocab  directory
    */
  def vocabDir = _vocabDir

  /**
    * Set vocab directory
    */
  def vocabDir_=(value: String) = _vocabDir = value

  /**
    * Get embedding dataframe directory
    */
  def embeddingDFDir = _embeddingDFDir

  /**
    * Set embedding dataframe directory
    */
  def embeddingDFDir_=(value: String) = _embeddingDFDir = value

  /**
   * Solver parameter
   */
  def solverParameter = {
    if (_solverParameter == null && _protoFile != null && _protoFile.length>0)
      protoFile = _protoFile
    _solverParameter
  }

  /**
   * Network parameter
   */
  def netParam = {
    if (_netParam == null && _protoFile != null && _protoFile.length>0)
      protoFile = _protoFile
    _netParam
  }

  def this(sc: SparkContext, args: Array[String]) {

    this(sc)
    //parse CLI arguments
    val cmd: CommandLine = {
      val options: Options = new Options
      options.addOption("conf", "conf", true, "solver configuration")
      options.addOption("train", "train", false, "training mode")
      options.addOption("test", "test", false, "test mode")
      options.addOption("features", "features", true, "name of output blobs")
      options.addOption("label", "label", true, "name of label blobs to be included in features")
      options.addOption("outputFormat", "outputFormat", true,
        "feature output format, currently support json and parquet, default: json")
      options.addOption("model", "model", true, "model path")
      options.addOption("output", "output", true, "output path")
      options.addOption("devices", "devices", true, "number of local GPUs")
      options.addOption("persistent", "persistent", false,
        "should data files be persistented on local file system?")
      options.addOption("snapshot", "snapshot", true, "snapshot state file path")
      options.addOption("weights", "weights", true, "snapshot model file path")
      options.addOption("connection", "connection", true, "ethernet or infiniband (default)")
      options.addOption("resize", "resize", false, "resize input image")
      options.addOption("clusterSize", "clusterSize", true, "size of the cluster")
      options.addOption("lmdb_partitions", "lmdb_partitions", true, "the # of LMDB RDD partitions. Default: cluster size")
      //used for dev purpose only
      options.addOption("imageRoot", "imageRoot", true, "image files' root")
      options.addOption("labelFile", "labelFile", true, "label file")
      options.addOption("captionFile", "captionFile", true, "caption file path")
      options.addOption("captionLength", "captionLength", true, "caption vector length")
      options.addOption("vocabSize", "vocabSize", true, "vocabulary size")
      options.addOption("imageCaptionDFDir", "imageCaptionDFDir", true, "Output Image Caption DataFrame Directory")
      options.addOption("vocabDir", "vocabDir", true, "Output Vocabulary Directory")
      options.addOption("embeddingDFDir", "embeddingDFDir", true, "Output Embedded DataFrame Directory")
      new BasicParser().parse(options, args)
    }

    protoFile = if (cmd.hasOption("conf")) cmd.getOptionValue("conf") else ""
    isTraining = cmd.hasOption("train")
    isTest = cmd.hasOption("test")
    isFeature = cmd.hasOption("features")
    label = if (cmd.hasOption("label")) cmd.getOptionValue("label") else ""
    modelPath = if (cmd.hasOption("model")) cmd.getOptionValue("model") else ""
    outputPath = if (cmd.hasOption("output")) cmd.getOptionValue("output") else "output"
    devices = if (cmd.hasOption("devices")) Integer.parseInt(cmd.getOptionValue("devices")) else 1
    isRddPersistent = cmd.hasOption("persistent")
    snapshotModelFile = if (cmd.hasOption("weights")) cmd.getOptionValue("weights") else ""
    snapshotStateFile = if (cmd.hasOption("snapshot")) cmd.getOptionValue("snapshot") else ""
    resize = cmd.hasOption("resize")

    connection = if (!cmd.hasOption("connection")) CaffeNet.RDMA
    else {
      val str: String = cmd.getOptionValue("connection")
      if (str.equalsIgnoreCase("ethernet")) CaffeNet.SOCKET
      else CaffeNet.RDMA
    }

    clusterSize = {
      val sparkMaster = if (sc == null) "" else sc.getConf.get("spark.master")
      if (sc.getConf.getBoolean("spark.dynamicAllocation.enabled", false)) {
        val maxExecutors = sc.getConf.getInt("spark.dynamicAllocation.maxExecutors", 1)
        val minExecutors = sc.getConf.getInt("spark.dynamicAllocation.minExecutors", 1)
        if (isTraining)
          assert(maxExecutors == minExecutors,
            "spark.dynamicAllocation.maxExecutors and spark.dynamicAllocation.minExecutors must be identical")
        minExecutors
      } else {
        if (sparkMaster.startsWith("yarn"))
          sc.getConf.getInt("spark.executor.instances", 1)
        else if (cmd.hasOption("clusterSize")) Integer.parseInt(cmd.getOptionValue("clusterSize"))
        else 1
      }
    }


    features =
      if (cmd.hasOption("features")) {
        val features_str = cmd.getOptionValue("features")
        features_str.split(",")
      } else {
        null
      }

    outputFormat = if (cmd.hasOption("outputFormat")) cmd.getOptionValue("outputFormat") else "json"

    lmdb_partitions = if (!cmd.hasOption("lmdb_partitions")) clusterSize
    else Integer.parseInt(cmd.getOptionValue("lmdb_partitions"))

    imageRoot = if (cmd.hasOption("imageRoot")) cmd.getOptionValue("imageRoot") else null
    labelFile = if (cmd.hasOption("labelFile")) cmd.getOptionValue("labelFile") else null
    imageCaptionDFDir = if (cmd.hasOption("imageCaptionDFDir")) cmd.getOptionValue("imageCaptionDFDir") else ""
    vocabDir = if (cmd.hasOption("vocabDir")) cmd.getOptionValue("vocabDir") else ""
    embeddingDFDir = if (cmd.hasOption("embeddingDFDir")) cmd.getOptionValue("embeddingDFDir") else ""
    captionFile = if (cmd.hasOption("captionFile")) cmd.getOptionValue("captionFile") else ""
    captionLength = if (cmd.hasOption("captionLength")) Integer.parseInt(cmd.getOptionValue("captionLength")) else 20
    vocabSize = if (cmd.hasOption("vocabSize")) Integer.parseInt(cmd.getOptionValue("vocabSize")) else -1

  }

  override def toString(): String = {
    val buildr: StringBuilder = new StringBuilder()
    buildr.append("protoFile:").append(protoFile).append("\n")
    buildr.append("train:").append(isTraining).append("\n")
    buildr.append("test:").append(isTest).append("\n")
    if (features != null)
      buildr.append("features:").append(features.mkString(",")).append("\n")
    buildr.append("label:").append(label).append("\n")
    buildr.append("outputFormat:").append(outputFormat).append("\n")
    buildr.append("model:").append(modelPath).append("\n")
    buildr.append("output:").append(outputPath).append("\n")
    buildr.append("devices:").append(devices).append("\n")
    buildr.append("persistent:").append(isRddPersistent).append("\n")
    buildr.append("snapshot:").append(snapshotStateFile).append("\n")
    buildr.append("weights:").append(snapshotModelFile).append("\n")
    buildr.append("clusterSize:").append(clusterSize).append("\n")
    buildr.append("lmdb_partitions:").append(lmdb_partitions).append("\n")
    buildr.append("train_data_layer_id:").append(train_data_layer_id).append("\n")
    buildr.append("test_data_layer_id:").append(test_data_layer_id).append("\n")
    buildr.append("transform_thread_per_device:").append(transform_thread_per_device).append("\n")
    buildr.append("captionFile:").append(captionFile).append("\n")
    buildr.append("captionLength:").append(captionLength).append("\n")
    buildr.append("vocabSize:").append(vocabSize).append("\n")
    buildr.append("imageCaptionDFDir:").append(imageCaptionDFDir).append("\n")
    buildr.append("vocabDir:").append(vocabDir).append("\n")
    buildr.append("embeddingDFDir:").append(embeddingDFDir).append("\n")
    buildr.toString()
  }
}

