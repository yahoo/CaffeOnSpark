// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe.tools

import java.io.{ObjectOutputStream, ByteArrayOutputStream}

import com.yahoo.ml.caffe.{LmdbRDD, Config, LMDB}
import org.apache.hadoop.io.BytesWritable
import org.apache.spark.{SparkContext, SparkConf}
import org.slf4j.{LoggerFactory, Logger}

object LMDB2Sequence {
  val log: Logger = LoggerFactory.getLogger(this.getClass)

  def main(args: Array[String]) {
    val sc_conf = new SparkConf()
    val sc = new SparkContext(sc_conf)

    //configure
    val conf = new Config(sc, args)
    if (conf.imageRoot.length == 0 || conf.outputPath.length == 0) {
      log.error("-imageRoot <LMDB directory> and -output <SequenceFile> must be defined")
      return
    }

    //produce RDD
    val rdd = new LmdbRDD(sc, conf.imageRoot, conf.lmdb_partitions).flatMap{
      case (id, label, channels, height, width, encoded, value) => {
        if (value == null) None
        else {
          val aout = new ByteArrayOutputStream
          val oos = new ObjectOutputStream(aout)
          oos.writeObject((id, label, channels, height, width, encoded))
          val tuple = (new BytesWritable(aout.toByteArray), new BytesWritable(value))
          aout.close()

          Some(tuple)
        }
      }
    }

    //save into output file on HDFS
    rdd.saveAsSequenceFile(conf.outputPath)
    sc.stop()
  }
}
