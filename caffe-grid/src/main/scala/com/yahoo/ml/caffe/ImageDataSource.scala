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
  extends DataSource[(String, String, Int, Int, Int, Boolean, Array[Byte]), (MatVector, FloatBlob)](conf,
    layerId, isTrain, (null, null, 0, 0, 0, false, null)) {
  @transient protected var log: Logger = null
  @transient protected var memdatalayer_param: MemoryDataParameter = null
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

    memdatalayer_param = layerParameter.getMemoryDataParam()
    numChannels = memdatalayer_param.getChannels()
    height = memdatalayer_param.getHeight()
    width = memdatalayer_param.getWidth()
    batchSize_ = memdatalayer_param.getBatchSize()
    if (batchSize_ < 1) {
      log.error("Invalid batch size:" + batchSize_)
      return false
    }
    log.info("Batch size:" + batchSize_)

    sourceFilePath = memdatalayer_param.getSource()
    if (sourceFilePath == null || sourceFilePath.isEmpty) {
      log.error("Source must be specified for layer " + layerId)
      return false
    }
    if (sourceFilePath.startsWith(FSUtils.localfsPrefix)) {
      val path = new URI(sourceFilePath).toString().substring(FSUtils.localfsPrefix.length)
      if (!path.startsWith("/")) {
        val f = new File(path)
        sourceFilePath = FSUtils.localfsPrefix+f.getAbsolutePath()
      }
    }

    true
  }

  /* make a data blob for solver/test threads */
  def dummyDataBlobs(): Array[FloatBlob] = {
    val dataBlobs: Array[FloatBlob] = Array()
    val labelBlob = new FloatBlob()
    labelBlob.reshape(batchSize_, 1, 1, 1)
    val dataBlob = new FloatBlob()
    dataBlob.reshape(batchSize_, numChannels, height, width)
    // last element is label blob
    dataBlobs :+ dataBlob :+ labelBlob
  }

  /* make a data holder for solver/test threads */
  def dummyDataHolder(): (MatVector, FloatBlob) = {
    val labelBlob = new FloatBlob()
    labelBlob.reshape(batchSize_, 1, 1, 1)
    val matVector = new MatVector(batchSize_)
    (matVector, labelBlob)
  }

  /* create a batch of samples extracted from source queue, up to a batch size.
   *
   * return false if seeing STOP_MARK from source queue
   * */
  def nextBatch(sampleIds: Array[String], data: (MatVector, FloatBlob)): Boolean = {
    val mats: MatVector = data._1
    val labels: FloatBlob = data._2
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
        val sample_id = sample._1
        val sample_label = sample._2
        val sample_channels = sample._3
        val sample_height = sample._4
        val sample_width = sample._5
        val sample_encoded = sample._6
        val sample_data = sample._7

        sampleIds(count) = sample_id
        labelCPU.set(count, sample_label.toInt)
        if (sample_height > 0 && sample_width > 0) {
          mat = new Mat(sample_channels, sample_height, sample_width, sample_data)
          if (sample_encoded)
            mat.decode(Mat.CV_LOAD_IMAGE_UNCHANGED)
          if (mat.width() != sample_width || mat.height() != sample_height) {
            log.warn("Skip image " + sample_id)
            mat = null
          }
          else if (conf.resize && ((sample_height != height) || (sample_width != width))) {
            log.info("Resize from " + sample_height + "x" + sample_width + " to " + height + "x" + width)
            mat.resize(height, width)
          }
        } else {
          mat = new Mat(sample_data)
          if (sample_encoded) {
            numChannels match {
              case 1 => mat.decode(Mat.CV_LOAD_IMAGE_GRAYSCALE)
              case 3 => mat.decode(Mat.CV_LOAD_IMAGE_COLOR)
              case _ => mat.decode(Mat.CV_LOAD_IMAGE_UNCHANGED)
            }
          }
          if (mat.width() == 0) {
            log.warn("Skipped image " + sample_id)
            mat = null
          }
          else if (conf.resize)
            mat.resize(height, width)
        }

        if (mat != null) {
          oldmat = mats.put(count, mat)
          if (oldmat != null)
            oldmat.deallocate()
          count = count + 1
        }
      }
    }

    shouldContinue
  }

}

