// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe.tools

import java.io.{ObjectOutputStream, ByteArrayOutputStream}

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataInputStream, Path}
import org.apache.hadoop.io.{SequenceFile, BytesWritable}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.slf4j.Logger
import org.slf4j.LoggerFactory

import com.yahoo.ml.caffe.Config

object Binary2Sequence {
  val log: Logger = LoggerFactory.getLogger(this.getClass)

  def main(args: Array[String]) {
    val sc_conf = new SparkConf()
    val sc = new SparkContext(sc_conf)
    val conf = new Config(sc, args)

    if (conf.imageRoot.length==0 || conf.labelFile.length==0) {
      log.error("Both -imageRoot and -labelFile must be defined")
      return
    }

    new Binary2Sequence(sc, conf).makeRDD().saveAsSequenceFile(conf.outputPath)

    sc.stop()
  }
}

class Binary2Sequence(@transient sc: SparkContext, conf: Config) extends Serializable {
  def makeRDD() : RDD[(BytesWritable, BytesWritable)] = {
    sc.textFile(conf.labelFile).mapPartitions{ iter => {
      val log: Logger = LoggerFactory.getLogger(this.getClass)

      val train_fs = new Path(conf.imageRoot).getFileSystem(new Configuration)
      val buffer = new Array[Byte](1024 * 1024)

      var file_count = 0
      iter.map { line => {
        val line_splits = line.split(" ")
        if (line_splits.length != 2) {
          log.error("Each line of label files must be 2")
          (null, null)
        } else {
          val filename = line_splits(0)
          val label = line_splits(1)
          val filelabel = (filename, label)

          val aout = new ByteArrayOutputStream
          val oos = new ObjectOutputStream(aout)
          oos.writeObject(filelabel)

          val bout = new ByteArrayOutputStream
          try {
            val in: FSDataInputStream = train_fs.open(new Path(conf.imageRoot + "/" + filename))
            var len = in.read(buffer, 0, buffer.length)
            while (len > 0) {
              bout.write(buffer, 0, len)
              len = in.read(buffer, 0, buffer.length)
            }

            in.close
            file_count = file_count + 1
            log.debug("file name:" + filename + " label:" + label)

            (new BytesWritable(aout.toByteArray), new BytesWritable(bout.toByteArray))
          } catch {
            case e: Exception => {
              log.warn(filename+" read exception, and will be skiped")
              (null, null)
            }
          } finally {
            if (aout != null) aout.close
            if (bout != null) bout.close
          }
        }
      }
      }
    }
    }
  }
}
