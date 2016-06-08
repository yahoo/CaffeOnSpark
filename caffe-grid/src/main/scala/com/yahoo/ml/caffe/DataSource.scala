// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import java.util.concurrent.ArrayBlockingQueue

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import caffe.Caffe._
import com.yahoo.ml.jcaffe._
import org.apache.spark.sql.SparkSession
import org.slf4j.{LoggerFactory, Logger}

/**
 * Base class for various data sources.
 *
 * Each subclass must have a constructor with the following signature: (conf: Config, layerId: Int, isTrain: Boolean).
 * This is required by CaffeOnSpark at startup.
 *
 * @param conf CaffeSpark configuration
 * @param layerId the layer index in the network protocol file
 * @param STOP_MARK stop mark to indicate source is exhausted
 * @tparam T1 class of entries extracted from RDD
 * @tparam T2 class of data blob in batch
 */
abstract class DataSource[T1, T2](val conf: Config, val layerId : Int, val isTrain: Boolean, val STOP_MARK: T1)
  extends Serializable {
  @transient private[caffe] var solverParameter: SolverParameter = null
  @transient private[caffe] var layerParameter: LayerParameter = null
  @transient private[caffe] var transformationParameter:TransformationParameter = null
  @transient var sourceQueue: ArrayBlockingQueue[T1] = null
  @transient protected var sourceFilePath : String = null
  @transient protected var batchSize_ : Int = -1
  @transient protected var solverMode: Int = -1

  /**
   * Construct a sample RDD using the Spark 2.X interface
   * @param ss spark session
   * @return RDD created from this source
   */
  def makeRDD(ss: SparkSession) : RDD[T1]

  /**
   * construct a sample RDD
   * @param sc spark context
   * @return RDD created from this source
   */
  @deprecated("Use makeRDD(ss: SparkSession)")
  def makeRDD(sc: SparkContext) : RDD[T1]


  /**
   *  initialization of a Source within a process
   *  @return true if successfully initialized
   */
  def init() : Boolean = {

    //solver parameter
    solverParameter = conf.solverParameter
    solverMode = solverParameter.getSolverMode().getNumber()

    //layer parameter
    layerParameter = conf.netParam.getLayer(layerId)

    //transformer parameter
    transformationParameter = layerParameter.getTransformParam()

    //source queue
    sourceQueue = new ArrayBlockingQueue[T1](1024)

    true
  }

  /**
   * adjust batch size
   * @param size the new batch size
   */
  def setBatchSize(size: Int) : Unit = {
    batchSize_ = size
  }

  /**
   * batch size
   */
  def batchSize() : Int = batchSize_

  /**
   * make a dummy data blob to be used by Solver threads
   * @return a dummy data blob
   */
  def dummyDataHolder() : T2

  /**
   * make a dummy data blob to be used by Solver threads
   * @return a dummy data blob
   */
  def dummyDataBlobs() : Array[FloatBlob]

  /**
   * create a batch of samples extracted from source queue
   *
   * This method is Invoked by Transformer thread.
   * You should extract samples from source queue, parse it and produce a batch.
   * @param sampleIds holder for sample Ids
   * @param data holder for data blob
   * @param labels holder for label blob
   * @return true if successful
    */
  def nextBatch(sampleIds: Array[String], data: T2, labels: FloatBlob) : Boolean
}

object DataSource extends Serializable {
  @transient private val log: Logger = LoggerFactory.getLogger(this.getClass)

  def getSource(conf : Config, isTraining: Boolean): DataSource[_,_] = {
    val layerId = if (isTraining) conf.train_data_layer_id else conf.test_data_layer_id
    log.info("Source data layer:"+layerId)
    val layerParameter = conf.netParam.getLayer(layerId)

    //get JVM class name
    var class_name : String = if (layerParameter.hasSourceClass()) layerParameter.getSourceClass() else null
    if (class_name == null) {
      val layerType = layerParameter.getType()
      log.error("source_class must be defined for input data layer:"+layerType)
      return null
    }

    //load JVM class
    val clz = Class.forName(class_name)
    if (clz == null) {
      log.error("failed to load class: "+class_name)
      return null
    }

    //locate a constructore of source class
    val constructor = clz.getConstructor(conf.getClass, java.lang.Integer.TYPE, java.lang.Boolean.TYPE)
    if (constructor == null) {
      log.error(class_name + " doesn't have constructor of required signature (conf: Config, layerId: Int, isTrain: Boolean)")
      System.exit(4)
    }

    val source : DataSource[_,_] = constructor.newInstance(conf,
      new Integer(layerId),
      new java.lang.Boolean(isTraining)).asInstanceOf[DataSource[_,_]]
    source.init()

    source
  }
}
