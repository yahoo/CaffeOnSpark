// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe.tools

import com.yahoo.ml.caffe.{Config, LMDB}
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
    val seq = LMDB.makeSequence(conf.imageRoot).map {
      case (key, value) => (new BytesWritable(key), new BytesWritable(value))
    }

    //save into output file on HDFS
    sc.parallelize(seq).saveAsSequenceFile(conf.outputPath)
    sc.stop()
  }
}


object LoadTest {
  val log: Logger = LoggerFactory.getLogger(this.getClass)

  def main(args: Array[String]) {
    val seq = LMDB.makeSequence(args(0))
    log.info("# of entries:"+seq.size)
  }
}