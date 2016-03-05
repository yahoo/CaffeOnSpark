// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import caffe.Caffe._
import com.yahoo.ml.jcaffe.{CaffeNet, Utils}

import org.apache.commons.cli.{BasicParser, CommandLine, Options}
import org.apache.spark.{SparkContext, SparkConf}
import org.slf4j.{LoggerFactory, Logger}

import scala.collection.mutable.Seq

/**
 * CaffeSpark configuration
 */
class Config(sc: SparkContext, args: Array[String]) extends Serializable {
  //parse CLI arguments
  @transient private val cmd: CommandLine = {
    val options: Options = new Options
    options.addOption("conf", "conf", true, "solver configuration")
    options.addOption("train", "train", false, "training mode")
    options.addOption("test", "test", false, "test mode")
    options.addOption("features", "features", true, "name of output blobs")
    options.addOption("label", "label", true, "name of label blobs to be included in features")
    options.addOption("inputFormat", "inputFormat", true,
      "input dataframe format, currently support json and parquet, default: json")
    options.addOption("select", "select", true, "Dataframe SQL statements. default: none")
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
    //used for dev purpose only
    options.addOption("clusterSize", "clusterSize", true, "size of the cluster")
    options.addOption("imageRoot", "imageRoot", true, "image files' root")
    options.addOption("labelFile", "labelFile", true, "label file")
    new BasicParser().parse(options, args)
  }

  /**
   * file location of solver protocol file
   */
  val protoFile : String = if (cmd.hasOption("conf")) cmd.getOptionValue("conf") else ""

  /**
   * flag indicate whether we want to perform model training or not
   */
  val isTraining : Boolean  = cmd.hasOption("train")
  /**
   * flag indicate whether we want to perform model test or not
   */
  var isTest : Boolean  = cmd.hasOption("test")
  /**
   * flag indicate whether we want to perform model training or not
   */
  val isFeature : Boolean = cmd.hasOption("features")
  /**
   * label blob name
   */
  val label : String = if (cmd.hasOption("label")) cmd.getOptionValue("label") else ""
  /**
   * HDFS path for model file
   */
  val modelPath : String = if (cmd.hasOption("model")) cmd.getOptionValue("model") else ""
  /**
   * HDFS path for test results
   */
  val outputPath : String = if (cmd.hasOption("output")) cmd.getOptionValue("output") else "output"
  /**
   * # of GPUs per executor
   */
  val devices : Int = if (cmd.hasOption("devices")) Integer.parseInt(cmd.getOptionValue("devices")) else 1
  /**
   * flag indicate whether training RDD should be persistent or not
   */
  val isRddPersistent : Boolean = cmd.hasOption("persistent")
  /**
   * HDFS path for snapshot states
   */
  val snapshotStateFile : String = if (cmd.hasOption("snapshot")) cmd.getOptionValue("snapshot") else ""
  /**
   * HDFS path for snapshot models
   */
  val snapshotModelFile : String = if (cmd.hasOption("weights"))  cmd.getOptionValue("weights") else ""
  /**
   * flag for resizing input image
   */
   val resize : Boolean = cmd.hasOption("resize")
  /**
   * network interface for connection among Spark executors
   */
  val connection : Int = if (!cmd.hasOption("connection"))  CaffeNet.RDMA
    else {
      val str : String = cmd.getOptionValue("connection")
      if (str.equalsIgnoreCase("ethernet")) CaffeNet.SOCKET
      else  CaffeNet.RDMA
    }
  /**
   * # of executors in the cluster
   */
  var clusterSize : Int = {
    val sparkMaster = if (sc == null) "" else sc.getConf.get("spark.master")
    if (sparkMaster.startsWith("yarn")) sc.getConf.getInt("spark.executor.instances", 1)
    else if (cmd.hasOption("clusterSize")) Integer.parseInt(cmd.getOptionValue("clusterSize"))
    else 1
  }

  /**
   * layer ID of training data source
   */
  var train_data_layer_id = -1
  /**
   * layer ID of training data source
   */
  var test_data_layer_id = -1
  /**
   * # of transformer threads per device
   */
  val transform_thread_per_device = 1

  /* blob names of feature output blobs */
  var features : Array[String] =
    if (cmd.hasOption("features")){
      val features_str = cmd.getOptionValue("features")
      features_str.split(",")
    } else {
      null
  }

  /**
   * Input dataframe format. json or parquet
   */
  var inputFormat : String = if (cmd.hasOption("inputFormat")) cmd.getOptionValue("inputFormat") else "json"
  /**
   * Output dataframe format. json or parquet
   */
  var outputFormat : String = if (cmd.hasOption("outputFormat")) cmd.getOptionValue("outputFormat") else "json"

  /* tool: input path */
  var imageRoot : String = if (cmd.hasOption("imageRoot")) cmd.getOptionValue("imageRoot") else null
  var labelFile : String = if (cmd.hasOption("labelFile")) cmd.getOptionValue("labelFile") else null

  @transient
  var solverParameter: SolverParameter = null
  @transient
  var netParam: NetParameter = null

  /**
   * Initialization of configuration
   * @return initialized configuration
   */
  def init() : Config = {
    val log = LoggerFactory.getLogger(this.getClass)

    //model path
    if (modelPath == null || modelPath.length() == 0) {
        log.error("modelPath is required")
        return null
    }

    //snapshotModelFile
    if (!snapshotStateFile.isEmpty && snapshotModelFile.isEmpty) {
      log.error("to resume training, please provide input model file")
      return null
    }

    // check test or feature
    // if both test and features are present, we only do features
    if (isTest && isFeature) {
      log.warn("both -test and -features are found, we will do features only, disabling test mode.")
      isTest = false
    }

    //solver parameter
    solverParameter = Utils.GetSolverParam(protoFile)
    netParam = Utils.GetNetParam(solverParameter.getNet())

    //set train_data_layer_id and test_data_layer_id
    var layerId = 0
    while (train_data_layer_id<0 || test_data_layer_id<0) {
      val layerParameter = netParam.getLayer(layerId)
      for (i <- 0 until layerParameter.getIncludeCount()) {
        if (layerParameter.getInclude(i).getPhase() == Phase.TRAIN)
          train_data_layer_id = layerId
        if (layerParameter.getInclude(i).getPhase() == Phase.TEST)
          test_data_layer_id = layerId
      }

      layerId += 1
    }

    //expand features with label
    if (features != null && label != null && !label.isEmpty() && !features.contains(label)) {
      features = features :+ label
    }

    return this
  }

  override def toString(): String = {
    val buildr:StringBuilder = new StringBuilder()
    buildr.append("protoFile:").append(protoFile).append("\n")
    buildr.append("train:").append(isTraining).append("\n")
    buildr.append("test:").append(isTest).append("\n")
    buildr.append("features:").append(features.mkString(",")).append("\n")
    buildr.append("label:").append(label).append("\n")
    buildr.append("inputFormat:").append(inputFormat).append("\n")
    buildr.append("outputFormat:").append(outputFormat).append("\n")
    buildr.append("model:").append(modelPath).append("\n")
    buildr.append("output:").append(outputPath).append("\n")
    buildr.append("devices:").append(devices).append("\n")
    buildr.append("persistent:").append(isRddPersistent).append("\n")
    buildr.append("snapshot:").append(snapshotStateFile).append("\n")
    buildr.append("weights:").append(snapshotModelFile).append("\n")
    buildr.append("clusterSize:").append(clusterSize).append("\n")
    buildr.append("train_data_layer_id:").append(train_data_layer_id).append("\n")
    buildr.append("test_data_layer_id:").append(test_data_layer_id).append("\n")
    buildr.append("transform_thread_per_device:").append(transform_thread_per_device).append("\n")
    buildr.toString()
  }
}

