// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import java.io.{FilenameFilter, File}

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.storage.StorageLevel
import org.fusesource.lmdbjni.Env
import org.slf4j.{LoggerFactory, Logger}

import scala.collection.mutable.Map
import scala.collection.mutable.ArrayBuffer

/**
 * LMDB is a built-in data source class for LMDB data source.
 * You could use this class for your LMDB data sources.
 *
 * @param conf CaffeSpark configuration
 * @param layerId the layer index in the network protocol file
 * @param isTrain
 */
class LMDB(conf: Config, layerId: Int, isTrain: Boolean) extends ImageDataSource(conf, layerId, isTrain) {
  
  var lmdbRDDRecords = Map[String, RDD[(String, String, Int, Int, Int, Boolean, Array[Byte])]]()
  
  override def makeRDD(sc: SparkContext): RDD[(String, String, Int, Int, Int, Boolean, Array[Byte])] = {
    //get a RDD
    return lmdbRDDRecords.getOrElseUpdate(
      sourceFilePath,
      new LmdbRDD(sc, sourceFilePath, conf.lmdb_partitions)
        .filter(isNotDummy)
        .persist(StorageLevel.DISK_ONLY)
    )
  }

  private def isNotDummy(item : (String, String, Int, Int, Int, Boolean, Array[Byte])): Boolean = {
    item._7 != null
  }
}
