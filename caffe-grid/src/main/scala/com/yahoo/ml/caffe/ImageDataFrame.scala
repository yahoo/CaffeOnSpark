// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.storage.StorageLevel

/**
 * ImageDataFrame is a built-in data source class using Spark dataframe format.
 *
 * ImageDataFrame expects dataframe with 2 required columns (lable:String, data:byte[]),
 * and 5 optional columns (id: String, channels :Int, height:Int, width:Int, encoded: Boolean).
 *
 * ImageDataFrame could be configured via the following MemoryDataLayer parameter:
 * (1) dataframe_column_select ... a collection of dataframe SQL selection statements
 * (ex. "sampleId as id", "abs(height) as height")
 * (2) image_encoded ... indicate whether image data are encoded or not. (default: false)
 * (3) dataframe_format ... Dataframe Format. (default: parquet)
 *
 * @param conf CaffeSpark configuration
 * @param layerId the layer index in the network protocol file
 * @param isTrain
 */
class ImageDataFrame(conf: Config, layerId: Int, isTrain: Boolean)
  extends ImageDataSource(conf, layerId, isTrain) {

  /* construct a sample RDD */
  def makeRDD(sc: SparkContext): RDD[(String, String, Int, Int, Int, Boolean, Array[Byte])] = {
    val sqlContext = new SQLContext(sc)
    //load DataFrame
    var reader = sqlContext.read
    if (memdatalayer_param.hasDataframeFormat())
      reader = reader.format(memdatalayer_param.getDataframeFormat())
    var df: DataFrame = reader.load(sourceFilePath)

    //select columns if specified
    if (memdatalayer_param.getDataframeColumnSelectCount() > 0) {
      val selects = memdatalayer_param.getDataframeColumnSelectList()

      import scala.collection.JavaConversions._
      df = df.selectExpr(selects.toList:_*)
    }

    //check optional columns
    val column_names : Array[String] = df.columns
    val has_id : Boolean = column_names.contains("id")
    val has_channels : Boolean = column_names.contains("channels")
    val has_height : Boolean = column_names.contains("height")
    val has_width : Boolean = column_names.contains("width")
    val has_encoded : Boolean = column_names.contains("encoded")

    //mapping each row to RDD tuple
    df.rdd.map(row => {
        var id: String = if (!has_id) "" else row.getAs[String]("id")
        var label: String = row.getAs[String]("label")
        val channels  : Int = if (!has_channels) 0 else row.getAs[Int]("channels")
        val height  : Int = if (!has_height) 0 else row.getAs[Int]("height")
        val width : Int = if (!has_width) 0 else row.getAs[Int]("width")
        val encoded : Boolean = if (!has_encoded) memdatalayer_param.getImageEncoded() else row.getAs[Boolean]("encoded")
        val data : Array[Byte] = row.getAs[Any]("data") match {
          case str: String => str.getBytes
          case arr: Array[Byte@unchecked] => arr
          case _ => {
            log.error("Unsupport value type")
            null
          }
        }
        (id, label, channels, height, width, encoded, data)
      }).persist(StorageLevel.DISK_ONLY)
  }
}
