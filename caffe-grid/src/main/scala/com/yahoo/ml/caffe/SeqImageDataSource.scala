// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import java.io.{ObjectInputStream, ByteArrayInputStream}

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
  def makeRDD(sc: SparkContext): RDD[(String, String, Int, Int, Int, Boolean, Array[Byte])] = {
    //we need to copy key/value since Hadoop reader reuses its object
    sc.sequenceFile(sourceFilePath, classOf[BytesWritable], classOf[BytesWritable], conf.clusterSize)
      .map { case (key, value) => {
      var id: String = null
      var label: String = null
      var channels : Int = 1
      var height : Int = 0
      var width : Int = 0
      var encoded : Boolean = false

      if (key != null) {
        val bis = new ByteArrayInputStream(key.copyBytes())
        val ois = new ObjectInputStream(bis)
        ois.readObject match {
          //case 1: encoded as java Pair
          case java_pair: com.yahoo.ml.dl.caffe.Pair[String@unchecked, String@unchecked] => {
            id = java_pair.first
            label = java_pair.second
            encoded = true
          }
          //case 2: encoded as Scala tuple
          case scala_pair2: Tuple2[String@unchecked, String@unchecked] => {
            id = scala_pair2._1
            label = scala_pair2._2
            encoded = true
          }
          //case 2: encoded as Scala tuple
          case scala_pair6: Tuple6[String@unchecked, String@unchecked,
            Int@unchecked, Int@unchecked, Int@unchecked, Boolean@unchecked] => {
            id = scala_pair6._1
            label = scala_pair6._2
            channels = scala_pair6._3
            height = scala_pair6._4
            width = scala_pair6._5
            encoded = scala_pair6._6
          }
        }
        ois.close()
        bis.close()
      }

      (id, label, channels, height, width, encoded, if (value != null) value.copyBytes() else null)
    }
    }
  }
}
