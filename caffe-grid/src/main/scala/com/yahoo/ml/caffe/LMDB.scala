// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel

/**
 * LMDB is a built-in data source class for LMDB data source.
 * You could use this class for your LMDB data sources.
 *
 * @param conf CaffeSpark configuration
 * @param layerId the layer index in the network protocol file
 * @param isTrain
 */
class LMDB(conf: Config, layerId: Int, isTrain: Boolean) extends ImageDataSource(conf, layerId, isTrain) {

  /**
   * Create an RDD using Spark 2.X interface
   * @param ss spark session
   * @return RDD created from this source
   */
  override def makeRDD(ss: SparkSession): RDD[(String, String, Int, Int, Int, Boolean, Array[Byte])] = {
    makeRDD(ss.sparkContext)
  }

  /**
   * Create an RDD using Spark 1.X interface
   * @param sc spark context
   * @return RDD created from this source
   */
  override def makeRDD(sc: SparkContext): RDD[(String, String, Int, Int, Int, Boolean, Array[Byte])] = {
    //create a RDD
    new LmdbRDD(sc, sourceFilePath, conf.lmdb_partitions)
      .filter(isNotDummy)
      .persist(StorageLevel.DISK_ONLY)
  }

  private def isNotDummy(item : (String, String, Int, Int, Int, Boolean, Array[Byte])): Boolean = {
    item._7 != null
  }
}
