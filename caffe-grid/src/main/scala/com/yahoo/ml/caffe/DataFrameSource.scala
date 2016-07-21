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
import org.apache.spark.sql.{DataFrame, SQLContext}
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
      val top = cosDataParam.getTop(i)
      val dataType: CoSDataParameter.DataType = top.getType()
      val name: String = top.getName()
      val channels: Int = top.getChannels()
      val width: Int = top.getWidth()
      val height: Int = top.getHeight()
      val transformParam: TransformationParameter =
        if (top.hasTransformParam()) top.getTransformParam() else null
      tops(i) = new Top(dataType, name, channels, width, height, transformParam)
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
    df.map(row => {
      val id: String = if (!has_id) "" else row.getAs[String]("id")
      val sample = new Array[Any](numTops)
      (0 until numTops).map(i => sample(i) = row.getAs[Any](topNames(i)))
      (id, sample)
    })
  }

  /* helper function to set up a float blob */
  def setFloatBlob(offset: Int, data: Array[Float], blob: FloatBlob): Unit = {
    if (data.length + offset > blob.count()) {
      throw new IllegalArgumentException("blob size is "
        + blob.count() + ", but total data length is "
        + (data.length + offset) + ".")
    }
    val blobCPU = blob.cpu_data()
    for (i <- 0 until data.length) {
      val index = offset + i
      blobCPU.set(index, data(i))
    }
  }

  /* help function to set up a matvector */
  def setMatVector(offset: Int, topidx: Int, data: Array[Byte],
                   mats: MatVector, encoded: Boolean): Unit = {
    var mat: Mat = null
    var oldmat: Mat = null
    val imageHeight = tops(topidx).height_
    val imageWidth = tops(topidx).width_
    val imageChannels = tops(topidx).channels_
    if (imageHeight > 0 && imageWidth > 0 && imageChannels > 0) {
      mat = new Mat(imageChannels, imageHeight, imageWidth, data)
      if (encoded)
        mat.decode(Mat.CV_LOAD_IMAGE_UNCHANGED)
      if (mat.width() != imageWidth || mat.height() != imageHeight) {
        log.warn("Skip image " + topidx + ", " + offset)
        mat = null
      }
    } else {
      mat = new Mat(data)
      if (encoded) {
        imageChannels match {
          case 1 => mat.decode(Mat.CV_LOAD_IMAGE_GRAYSCALE)
          case 3 => mat.decode(Mat.CV_LOAD_IMAGE_COLOR)
          case _ => mat.decode(Mat.CV_LOAD_IMAGE_UNCHANGED)
        }
      }
      if (mat.width() == 0) {
        log.warn("Skipped image " + + topidx + ", " + offset)
        mat = null
      }
    }

    if (mat != null) {
      oldmat = mats.put(offset, mat)
      if (oldmat != null)
        oldmat.deallocate()
    }
  }

  /* make a data holder for solver/test threads */
  def dummyDataHolder(): Array[Any] = {
    val holder = new Array[Any](numTops)
    for (i <- 0 until numTops) {
      val dataType = getTopDataType(i)
      val channels = getTopChannel(i)
      val height = getTopHeight(i)
      val width = getTopWidth(i)
      holder(i) = dataType match {
        case CoSDataParameter.DataType.STRING | 
             CoSDataParameter.DataType.INT | 
             CoSDataParameter.DataType.FLOAT | 
             CoSDataParameter.DataType.INT_ARRAY | 
             CoSDataParameter.DataType.FLOAT_ARRAY => {
          val blob = new FloatBlob()
          blob.reshape(batchSize, channels, height, width)
          blob
        }
        case CoSDataParameter.DataType.RAW_IMAGE |
             CoSDataParameter.DataType.ENCODED_IMAGE => 
             new MatVector(batchSize)
      }
    }
    holder
  }

  def dummyDataBlobs(): Array[FloatBlob] = {
    val blob = new Array[FloatBlob](numTops)
    for (i <- 0 until numTops) {
      val channels = getTopChannel(i)
      val height = getTopHeight(i)
      val width = getTopWidth(i)
      blob(i) = new FloatBlob()
      blob(i).reshape(batchSize_, channels, height, width)
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
              setFloatBlob(offset, data, blob)
            }
            case CoSDataParameter.DataType.INT => {
              val data: Array[Float] = Array(
                sample._2(i).asInstanceOf[Int].toFloat)
              val blob: FloatBlob = batchData(i).asInstanceOf[FloatBlob]
              val offset: Int = count
              setFloatBlob(offset, data, blob)
            }
            case CoSDataParameter.DataType.FLOAT => {
              val data: Array[Float] = Array(sample._2(i).asInstanceOf[Float])
              val blob: FloatBlob = batchData(i).asInstanceOf[FloatBlob]
              val offset: Int = count
              setFloatBlob(offset, data, blob)
            }
            case CoSDataParameter.DataType.INT_ARRAY => {
              val data: WrappedArray[Int] = sample._2(
                i).asInstanceOf[WrappedArray[Int]]
              val floatData: Array[Float] = data.toArray.map(x => x.toFloat)
              val blob: FloatBlob = batchData(i).asInstanceOf[FloatBlob]
              val offset = count * tops(i).getSize()
              setFloatBlob(offset, floatData, blob)
            }
            case CoSDataParameter.DataType.FLOAT_ARRAY => {
              val data: Array[Float] = sample._2(
                i).asInstanceOf[WrappedArray[Float]].toArray
              val blob: FloatBlob = batchData(i).asInstanceOf[FloatBlob]
              val offset = count * tops(i).getSize()
              setFloatBlob(offset, data, blob)
            }
            case CoSDataParameter.DataType.ENCODED_IMAGE => {
              val data: Array[Byte] = sample._2(i).asInstanceOf[Array[Byte]]
              val mats: MatVector = batchData(i).asInstanceOf[MatVector]
              val offset: Int = count
              setMatVector(offset, i, data, mats, true)
            }
            case CoSDataParameter.DataType.RAW_IMAGE => {
              val data: Array[Byte] = sample._2(i).asInstanceOf[Array[Byte]]
              val mats: MatVector = batchData(i).asInstanceOf[MatVector]
              val offset: Int = count
              setMatVector(offset, i, data, mats, false)
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

  override def getTopHeight(index: Int): Int = tops(index).height_

  override def getTopWidth(index: Int): Int = tops(index).width_

  override def getTopChannel(index: Int): Int = tops(index).channels_
}


class Top(dataType:CoSDataParameter.DataType, name:String, channels:Int, 
          width:Int, height:Int, transformParam: TransformationParameter) {
  val dataType_ :CoSDataParameter.DataType = dataType
  val name_ :String =  name
  val channels_ :Int = channels
  val width_ :Int = width
  val height_ :Int = height
  val transformParam_ :TransformationParameter = transformParam

  def getSize() : Int = {
    return width_ * height_ * channels_
  }

  override def toString = s"Top(type=$dataType_, name=$name_, " +
    s"channels=$channels_, width=$width_, height=$height_)"
}
