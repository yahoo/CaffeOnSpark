// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.storage.StorageLevel
import org.slf4j.{LoggerFactory, Logger}

/**
 * ImageDataFrame is a built-in data source class using data frame format.
 * Each entry of sequence is a trip
 * (id: String, label: String, Int channels, Int height, Int width, data : byte[]).
 *
 * @param conf CaffeSpark configuration
 * @param layerId the layer index in the network protocol file
 * @param isTrain
 */
class ImageDataFrame(conf: Config, layerId: Int, isTrain: Boolean)
  extends ImageDataSource(conf, layerId, isTrain) {

  private def getValueAsBytes(value : Any) = {
    val seq: Array[Byte] =value match {
      case str: String => str.getBytes
      case arr: Array[Byte@unchecked] => arr
      case _ => {
        log.error("Unsupport value type")
        null
      }
    }

    if (seq == null) null else seq.clone()
  }

  /* construct a sample RDD */
  def makeRDD(sc: SparkContext): RDD[(String, String, Int, Int, Int, Boolean, Array[Byte])] = {
    //Data Frame
    val sqlContext = new SQLContext(sc)
    val df: DataFrame = sqlContext.read.format(conf.inputFormat).load(sourceFilePath)

    val rdd: RDD[(String, String, Int, Int, Int, Boolean, Array[Byte])] =
      if (conf.channelsExpr == null || conf.channelsExpr.isEmpty
        || conf.heightExpr == null || conf.heightExpr.isEmpty
        || conf.widthExpr == null || conf.widthExpr.isEmpty
        || conf.encodedExpr == null || conf.encodedExpr.isEmpty) {
        df.selectExpr(conf.idExpr, conf.labelExpr, conf.valueExpr)
          .map(row => {
          var id: String = if (row.isNullAt(0)) "" else row.getAs[String](0)
          var label: String = row.getAs[String](1)
          val data: Array[Byte] = getValueAsBytes(row(2))

          (id, label, 1, 0, 0, false, data)
        })
      } else {
        df.selectExpr(conf.idExpr, conf.labelExpr,
          conf.channelsExpr, conf.heightExpr, conf.widthExpr,
          conf.encodedExpr, conf.valueExpr)
          .map(row => {
          var id: String = if (row.isNullAt(0)) "" else row.getAs[String](0)
          var label: String = row.getAs[String](1)
          val channels = if (row.isNullAt(2)) 1 else row.getInt(2)
          val height = if (row.isNullAt(3)) 1 else row.getInt(3)
          val width = if (row.isNullAt(4)) 1 else row.getInt(4)
          val encoded = if (row.isNullAt(5)) false else row.getBoolean(5)
          val data: Array[Byte] = getValueAsBytes(row(6))

          (id, label, channels, height, width, encoded, data)
        })
      }

    rdd.persist(StorageLevel.DISK_ONLY)
  }
}

/**
 * ImageDataFrame using Parquet data frame format
 *
 * @param conf CaffeSpark configuration
 * @param layerId the layer index in the network protocol file
 * @param isTrain
 */
class ImageDataFrameParquet(conf: Config, layerId: Int, isTrain: Boolean)
  extends ImageDataFrame(conf, layerId, isTrain) {
  conf.inputFormat = "parquet"
}

/**
 * ImageDataFrame using Json data frame format
 *
 * @param conf CaffeSpark configuration
 * @param layerId the layer index in the network protocol file
 * @param isTrain
 */
class ImageDataFrameJson(conf: Config, layerId: Int, isTrain: Boolean)
  extends ImageDataFrame(conf, layerId, isTrain) {
  conf.inputFormat = "json"
}
