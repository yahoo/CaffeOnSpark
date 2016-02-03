// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import java.io.{File, ObjectInputStream, ByteArrayInputStream}
import java.net.URI

import caffe.Caffe._
import com.yahoo.ml.jcaffe._

import org.apache.hadoop.io.BytesWritable
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import org.slf4j.{LoggerFactory, Logger}


/**
 * abstract image data source for images.
 * Its subclass should construct RDD of  ((id: String, label: String), value : byte[]).
 *
 * @param conf CaffeSpark configuration
 * @param layerId the layer index in the network protocol file
 * @param isTrain
 */
abstract class ImageDataSource(conf: Config, layerId: Int, isTrain: Boolean)
  extends DataSource[(Array[Byte], Array[Byte]), MatVector](conf, layerId, isTrain, (null, null)) {
  @transient private var log: Logger = null
  @transient private var numChannels = 0
  @transient private var height = 0
  @transient private var width = 0

  /* initialization of an object within a JVM*/
  override def init(): Boolean = {
    log = LoggerFactory.getLogger(this.getClass)
    if (!super.init()) {
      log.error("Initialization failed in DataSource.init()")
      return false
    }

    if (!layerParameter.hasMemoryDataParam()) {
      log.error("Layer " + layerId + " failed to specify memory_data_param")
      return false
    }

    val mem_data_param: MemoryDataParameter = layerParameter.getMemoryDataParam()
    numChannels = mem_data_param.getChannels()
    height = mem_data_param.getHeight()
    width = mem_data_param.getWidth()
    batchSize_ = mem_data_param.getBatchSize()
    if (batchSize_ < 1) {
      log.error("Invalid batch size:" + batchSize_)
      return false
    }
    log.info("Batch size:" + batchSize_)

    sourceFilePath = mem_data_param.getSource()
    if (sourceFilePath == null || sourceFilePath.isEmpty) {
      log.error("Source must be specified for layer " + layerId)
      return false
    }
    if (sourceFilePath.startsWith(FSUtils.localfsPrefix)) {
      val path = new URI(sourceFilePath).toString().substring(FSUtils.localfsPrefix.length)
      if (!path.startsWith("/")) {
        val f = new File(path)
        sourceFilePath = f.getAbsolutePath()
      }
    }

    true
  }

  /* make a data blob for solver/test threads */
  def dummyDataBlobs(): Array[FloatBlob] = {
    val dataBlobs: Array[FloatBlob] = Array()
    val dataBlob = new FloatBlob()
    dataBlob.reshape(batchSize_, numChannels, height, width)
    dataBlobs :+ dataBlob
  }

  /* make a data holder for solver/test threads */
  def dummyDataHolder(): MatVector = {
    new MatVector(batchSize_)
  }

  /* create a batch of samples extracted from source queue, up to a batch size.
   *
   * return false if seeing STOP_MARK from source queue
   * */
  def nextBatch(sampleIds: Array[String], mats: MatVector, labels: FloatBlob): Boolean = {
    var labelCPU = labels.cpu_data()

    var shouldContinue = true
    var mat: Mat = null
    var oldmat: Mat = null
    var count: Int = 0
    while (count < batchSize_ && shouldContinue) {
      val sample = sourceQueue.take()

      if (sample == STOP_MARK) {
        log.info("Completed all files")
        shouldContinue = false
      } else {
        val key = sample._1
        val value = sample._2
        if (key != null && value != null) {
          val bis = new ByteArrayInputStream(key)
          val ois = new ObjectInputStream(bis)
          val file_label: (String, String) = ois.readObject match {
            //case 1: encoded as java Pair
            case java_pair: com.yahoo.ml.dl.caffe.Pair[String @unchecked, String @unchecked] => (java_pair.first, java_pair.second)
            //case 2: encoded as Scala tuple
            case scala_pair: Tuple2[String @unchecked, String @unchecked] => scala_pair
            //other
            case _ => {
              log.error("Unsupported data format for labels")
              null
            }
          }
          ois.close()
          bis.close()

          sampleIds(count) = file_label._1
          labelCPU.set(count, file_label._2.toInt)
          numChannels match {
            case 1 => {
              //Monochro
              mat = new Mat(height, width, value, false)
            }
            case 3 => {
              //Color
              mat = new Mat(value, false)
              mat.decode(Mat.CV_LOAD_IMAGE_COLOR)
              if (mat.width() == 0)
                log.warn("Skipped image " + file_label._1)
              else if (conf.resize)
                mat.resize(height, width)
            }
            case _ => {
              log.error("number of channel needs to be 1 or 3")
              mat = null
            }
          }
          if (mat != null) {
            oldmat = mats.put(count, mat)
            if (oldmat != null)
              oldmat.deallocate()
          }

          count = count + 1
        }
      }
    }

    shouldContinue
  }
}

