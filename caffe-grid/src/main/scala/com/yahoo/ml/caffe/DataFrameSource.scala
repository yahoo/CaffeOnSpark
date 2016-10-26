// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import java.io.File
import java.net.URI

import caffe.Caffe._
import com.yahoo.ml.jcaffe._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, DataFrame, SQLContext}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.WrappedArray

/**
 * Rewrite
 * DataFrameSource is a built-in data source class using Spark dataframe format.
 *
 * It is coupled with Caffe CoSDataLayer. The user specifies the top blobs in
 * that layer.
 *
 * @param conf CaffeSpark configuration
 * @param layerId the layer index in the network protocol file
 * @param isTrain
*/

class DataFrameSource(conf: Config, layerId: Int, isTrain: Boolean)
  extends DataSource[(String, Array[Any]), Array[Any]](conf,
    layerId, isTrain, ("", Array[Any]())) {
  @transient protected var log: Logger = null
  @transient protected var cosDataParam: CoSDataParameter = null
  @transient private var tops: Array[Top] = null
  private var numTops: Int = -1

  /* initialization of an object within a JVM*/
  override def init(): Boolean = {
    log = LoggerFactory.getLogger(this.getClass)
    if (!super.init()) {
      log.error("Initialization failed in DataSource.init()")
      return false
    }

    if (!layerParameter.hasCosDataParam()) {
      log.error("Layer " + layerId + " failed to specify cos_data_param")
      return false
    }

    cosDataParam = layerParameter.getCosDataParam()
    numTops = cosDataParam.getTopCount()
    tops = new Array[Top](numTops)
    for (i <- 0 until numTops) {
      val topParam = cosDataParam.getTop(i)
      tops(i) = new Top(topParam)
    }

    batchSize_ = cosDataParam.getBatchSize()
    log.info("Batch size:" + batchSize_)

    sourceFilePath = cosDataParam.getSource()
    if (sourceFilePath == null || sourceFilePath.isEmpty) {
      log.error("Source must be specified for layer " + layerId)
      return false
    }
    if (sourceFilePath.startsWith(FSUtils.localfsPrefix)) {
      val path = new URI(sourceFilePath).toString()
        .substring(FSUtils.localfsPrefix.length)
      if (!path.startsWith("/")) {
        val f = new File(path)
        sourceFilePath = FSUtils.localfsPrefix+f.getAbsolutePath()
      }
    }

    true
  }

  /* construct a sample RDD */
  def makeRDD(sc: SparkContext): RDD[(String, Array[Any])] = {
    val sqlContext = new SQLContext(sc)
    //load DataFrame, default parquet format
    var reader = sqlContext.read
    if (cosDataParam.hasDataframeFormat())
      reader = reader.format(cosDataParam.getDataframeFormat())
    else
      reader = reader.format("parquet")
    var df: DataFrame = reader.load(sourceFilePath)

    //check data columns
    val columnNames : Array[String] = df.columns
    val topNames: Array[String] = new Array(numTops)
    for (i <- 0 until numTops) {
      topNames(i) = tops(i).name_
      require(columnNames.contains(topNames(i)),
        s"$topNames(i) does not exist in dataframe")
    }
    val has_id : Boolean = columnNames.contains("id")

    //mapping each row to RDD tuple
    df.rdd.map(row => {
      val id: String = if (!has_id) "" else row.getAs[String]("id")
      val sample = new Array[Any](numTops)
      (0 until numTops).map(i => sample(i) = row.getAs[Any](topNames(i)))
      (id, sample)
    })
  }

  /* helper function to set up a float blob */
  def setFloatBlob(offset: Int, stride: Int, data: Array[Float],
                   blob: FloatBlob): Unit = {
    val dataLen: Int = (data.length - 1) * stride + offset + 1
    if (dataLen > blob.count()) {
      throw new IllegalArgumentException("blob size is "
        + blob.count() + ", but total data length is "
        + dataLen + ".")
    }
    val blobCPU = blob.cpu_data()
    for (i <- 0 until data.length) {
      val index = offset + i * stride
      blobCPU.set(index, data(i))
    }
  }

  /* help function to set up a matvector */
  def setMatVector(offset: Int, topidx: Int, data: Array[Byte], rawHeight: Int,
                   rawWidth: Int, mats: MatVector, encoded: Boolean): Unit = {
    var mat: Mat = null
    var oldmat: Mat = null
    val imageHeight = tops(topidx).height_
    val imageWidth = tops(topidx).width_
    val imageChannels = tops(topidx).channels_
    val rawDim: Boolean = (rawHeight > 0) && (rawWidth > 0)

    if (encoded) {
      // encoded image
      mat = new Mat(data)
      imageChannels match {
        case 1 => mat.decode(Mat.CV_LOAD_IMAGE_GRAYSCALE)
        case 3 => mat.decode(Mat.CV_LOAD_IMAGE_COLOR)
        case _ => mat.decode(Mat.CV_LOAD_IMAGE_UNCHANGED)
      }
      if (rawDim && ((mat.height() != rawHeight) ||
            (mat.width() != rawWidth))) {
        // raw dimension provided but mat does not match them
        log.warn("Skip image at top#" + topidx + ", index#" + offset)
        mat = null
      }
    } else {
      // raw image
      if (rawDim) {
        // raw image provided
        mat = new Mat(imageChannels, rawHeight, rawWidth, data)
        if (mat.width() != rawWidth || mat.height() != rawHeight) {
          log.warn("Skip image at top#" + topidx + ", index#" + offset)
          mat = null
        }
      } else {
        // raw image not provided, use dim from proto instead
        mat = new Mat(imageChannels, imageHeight, imageWidth, data)
        if (mat.width() != imageWidth || mat.height() != imageHeight) {
          log.warn("Skip image at top#" + topidx + ", index#" + offset)
          mat = null
        }
      }
    }
    if (mat != null) {
      if (conf.resize && ((mat.height() != imageHeight) ||
        (mat.width() != imageWidth))) {
        mat.resize(imageHeight, imageWidth)
      }
      oldmat = mats.put(offset, mat)
      if (oldmat != null)
        oldmat.deallocate()
    }
  }

  /* make a data holder for solver/test threads */
  def dummyDataHolder(): Array[Any] = {
    val holder = new Array[Any](numTops)
    for (i <- 0 until numTops) {
      val dataType = tops(i).dataType_
      val sampleShape = tops(i).sampleShape_
      val transpose = tops(i).transpose_
      // if transpose is on, blobShape has exactly 2 axes.
      val blobShape = if (transpose) Array(sampleShape(0), batchSize_)
        else batchSize_ +: sampleShape
      holder(i) = dataType match {
        case CoSDataParameter.DataType.STRING | 
             CoSDataParameter.DataType.INT | 
             CoSDataParameter.DataType.FLOAT | 
             CoSDataParameter.DataType.INT_ARRAY | 
             CoSDataParameter.DataType.FLOAT_ARRAY => {
          val blob = new FloatBlob()
          blob.reshape(blobShape)
          blob
        }
        case CoSDataParameter.DataType.RAW_IMAGE |
             CoSDataParameter.DataType.ENCODED_IMAGE |
             CoSDataParameter.DataType.ENCODED_IMAGE_WITH_DIM =>
             new MatVector(batchSize_)
      }
    }
    holder
  }

  def dummyDataBlobs(): Array[FloatBlob] = {
    val blob = new Array[FloatBlob](numTops)
    for (i <- 0 until numTops) {
      val sampleShape = tops(i).outSampleShape_
      val transpose = tops(i).transpose_
      // if transpose is on, blobShape has exactly 2 axes.
      val blobShape = if (transpose) Array(sampleShape(0), batchSize_)
        else batchSize_ +: sampleShape
      blob(i) = new FloatBlob()
      blob(i).reshape(blobShape)
    }
    blob
  }

  /* create a batch of samples extracted from source queue, up to a batch size.
 *
 * return false if seeing STOP_MARK from source queue
 * */
  def nextBatch(batchIds: Array[String], batchData: Array[Any]): Boolean = {
    var shouldContinue = true
    var count: Int = 0
    while (count < batchSize_ && shouldContinue) {
      val sample = sourceQueue.take()
      if (sample == STOP_MARK) {
        log.info("Completed all files")
        shouldContinue = false
      } else {
        for (i <- 0 until numTops) {
          batchIds(count) = sample._1
          val dataType = tops(i).dataType_ match {
            case CoSDataParameter.DataType.STRING => {
              val data: Array[Float] = Array(
                sample._2(i).asInstanceOf[String].toFloat)
              val blob: FloatBlob = batchData(i).asInstanceOf[FloatBlob]
              val offset: Int = count
              setFloatBlob(offset, 1, data, blob)
            }
            case CoSDataParameter.DataType.INT => {
              val data: Array[Float] = Array(
                sample._2(i).asInstanceOf[Int].toFloat)
              val blob: FloatBlob = batchData(i).asInstanceOf[FloatBlob]
              val offset: Int = count
              setFloatBlob(offset, 1, data, blob)
            }
            case CoSDataParameter.DataType.FLOAT => {
              val data: Array[Float] = Array(sample._2(i).asInstanceOf[Float])
              val blob: FloatBlob = batchData(i).asInstanceOf[FloatBlob]
              val offset: Int = count
              setFloatBlob(offset, 1, data, blob)
            }
            case CoSDataParameter.DataType.INT_ARRAY => {
              val data: WrappedArray[Int] = sample._2(
                i).asInstanceOf[WrappedArray[Int]]
              val floatData: Array[Float] = data.toArray.map(x => x.toFloat)
              val blob: FloatBlob = batchData(i).asInstanceOf[FloatBlob]
              val offset = if (tops(i).transpose_) count else count * tops(i).size_
              val stride = if (tops(i).transpose_) batchSize_ else 1
              setFloatBlob(offset, stride, floatData, blob)
            }
            case CoSDataParameter.DataType.FLOAT_ARRAY => {
              val data: Array[Float] = sample._2(
                i).asInstanceOf[WrappedArray[Float]].toArray
              val blob: FloatBlob = batchData(i).asInstanceOf[FloatBlob]
              val offset = if (tops(i).transpose_) count else count * tops(i).size_
              val stride = if (tops(i).transpose_) batchSize_ else 1
              setFloatBlob(offset, stride, data, blob)
            }
            case CoSDataParameter.DataType.ENCODED_IMAGE => {
              val data: Array[Byte] = sample._2(i).asInstanceOf[Array[Byte]]
              val mats: MatVector = batchData(i).asInstanceOf[MatVector]
              val offset: Int = count
              setMatVector(offset, i, data, -1, -1, mats, true)
            }
            case CoSDataParameter.DataType.RAW_IMAGE => {
              val data: Array[Byte] = sample._2(i).asInstanceOf[Array[Byte]]
              val mats: MatVector = batchData(i).asInstanceOf[MatVector]
              val offset: Int = count
              setMatVector(offset, i, data, -1, -1, mats, false)
            }
            case CoSDataParameter.DataType.ENCODED_IMAGE_WITH_DIM => {
              val row: Row = sample._2(i).asInstanceOf[Row]
              val data: Array[Byte] = row.getAs[WrappedArray[Byte]]("image").toArray
              val height: Int = row.getAs[Int]("height")
              val width: Int = row.getAs[Int]("width")
              val mats: MatVector = batchData(i).asInstanceOf[MatVector]
              val offset: Int = count
              setMatVector(offset, i, data, height, width, mats, true)
            }
            case _ => throw new Exception("Unsupported data type for CoS Data Layer")
          }
        }
      }
      count += 1
    }
    shouldContinue
  }

  override def useCoSDataLayer(): Boolean = true

  override def getNumTops(): Int = numTops

  override def getTopDataType(index: Int): CoSDataParameter.DataType 
       = tops(index).dataType_

  override def getTopTransformParam(index: Int): TransformationParameter
       = tops(index).transformParam_
}

class Top(topParam: CoSDataParameter.TopBlob) {
  val dataType_ :CoSDataParameter.DataType = topParam.getType()
  val name_ :String =  topParam.getName()
  // pre-transformer (if exists) blob dimensions
  val channels_ :Int = topParam.getChannels()
  val width_ :Int = topParam.getWidth()
  val height_ :Int = topParam.getHeight()
  val sampleNumAxes_ :Int = topParam.getSampleNumAxes()
  // post-transformer (if exists) blob dimensions
  val outChannels_ :Int = if (topParam.getOutChannels() == 0) channels_
                           else topParam.getOutChannels()
  val outHeight_ :Int = if (topParam.getOutHeight() == 0) height_
                        else topParam.getOutHeight()
  val outWidth_ :Int = if (topParam.getOutWidth() == 0) width_
                       else topParam.getOutWidth()
  val transformParam_ :TransformationParameter =
    if (topParam.hasTransformParam()) topParam.getTransformParam() else null
  val shape_ :Array[Int] = Array(channels_, height_, width_)
  val outShape_ :Array[Int] = Array(outChannels_, outHeight_, outWidth_)
  // pre-transformer (if exists) blob shape
  val sampleShape_ :Array[Int] = shape_.slice(0,sampleNumAxes_)
  // post-transformer (if exists) blob shape
  val outSampleShape_ :Array[Int] = outShape_.slice(0,sampleNumAxes_)
  // pre-transformer (if exists) blob shape
  val size_ :Int = sampleShape_.foldLeft(1)(_*_)
  // post-transformer (if exists) blob shape
  val outSize_ :Int = outSampleShape_.foldLeft(1)(_*_)
  // flag to transpose the blob
  val transpose_ :Boolean = topParam.getTranspose()
  if (transpose_) {
    assert(sampleNumAxes_ == 1)
  }


  override def toString = s"Top(type=$dataType_, name=$name_, " +
    s"channels=$channels_, height=$height_, width=$width_, " +
    s"outChannels=$outChannels_, outHeight=$outHeight_, outWidth=$outWidth_, " +
    s"sampleNumAxes=$sampleNumAxes_, size=$size_, outSize=$outSize_)"
}
