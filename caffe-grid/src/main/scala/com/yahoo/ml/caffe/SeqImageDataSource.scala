// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import org.apache.hadoop.io.BytesWritable
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

/**
 * SeqImageDataSource is a built-in data source class using sequence file format.
 * Each entry of sequence is a tuple ((id: String, label: String), value : byte[]).
 *
 * @param conf CaffeSpark configuration
 * @param layerId the layer index in the network protocol file
 * @param isTrain
 */
class SeqImageDataSource(conf: Config, layerId: Int, isTrain: Boolean)
  extends ImageDataSource(conf, layerId, isTrain) {

  /* construct a sample RDD */
  def makeRDD(sc: SparkContext): RDD[(Array[Byte], Array[Byte])] = {
    //we need to copy key/value since Hadoop reader reuses its object
    sc.sequenceFile(sourceFilePath, classOf[BytesWritable], classOf[BytesWritable], conf.clusterSize)
      .map { case (key, value) => (if (key != null) key.copyBytes else null,
      if (value != null) value.copyBytes else null)
    }
  }
}

