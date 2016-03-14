// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import java.io.{FilenameFilter, File}

import caffe.Caffe.Datum
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.rdd.RDD
import org.apache.spark.{Logging, Partition, SparkContext, TaskContext}
import org.fusesource.lmdbjni.{Transaction, Database, Env}
import org.slf4j.{LoggerFactory, Logger}

import scala.collection.mutable

/**
 * Each LMDB RDD partition has a start key. from which we will enumerate a number of entries
 * @param idx partition ID
 * @param startKey start key
 * @param size # of entries in this partition
 */
private[caffe] class LmdbPartition(idx: Int, val startKey: Array[Byte], val size: Int) extends Partition {
  override def index: Int = idx
}

/**
 * LmdbRDD is a custom RDD for accessing LMDB databases using a specified # of partitions.
 *
 * @param sc spark context
 * @param lmdb_path URI of LMDB databases
 * @param numPartitions # of the desired partitions.
 */

class LmdbRDD(@transient val sc: SparkContext, val lmdb_path: String, val numPartitions: Int)
  extends RDD[(String, String, Int, Int, Int, Boolean, Array[Byte])](sc, Nil) with Logging {

  override def getPartitions: Array[Partition] = {
    //load lmdbjni
    LmdbRDD.loadLibrary()

    var part_index: Int = 0
    var pos: Int = 0
    val env: Env = new Env(LmdbRDD.toLocalFile(lmdb_path))
    val db: Database = env.openDatabase(null, 0)
    val size: Long = db.stat().ms_entries
    val part_size: Int = Math.ceil(size.toDouble / numPartitions.toDouble).toInt

    val partitions = new Array[Partition](numPartitions)
    val txn = env.createReadTransaction()
    try {
      val it = db.iterate(txn)
      while (it.hasNext && part_index < numPartitions) {
        val next = it.next()
        val key: Array[Byte] = next.getKey()

        if (pos % part_size == 0) {
          partitions(part_index) = new LmdbPartition(part_index, key, part_size)
          part_index = part_index + 1
        }

        pos = pos + 1
      }
    } catch {
      case e: Exception =>
        logWarning(e.toString)
    } finally {
      doCommit(txn)
      doClose(db)
    }

    logInfo(partitions.length + " LMDB RDD partitions")
    partitions
  }

  override def compute(thePart: Partition, context: TaskContext):
  Iterator[(String, String, Int, Int, Int, Boolean, Array[Byte])] =
    new Iterator[(String, String, Int, Int, Int, Boolean, Array[Byte])] {
      logInfo("Processing partition " + thePart.index)
      //load lmdbjni
      LmdbRDD.loadLibrary()

      //create an iterator
      val env: Env = new Env(LmdbRDD.toLocalFile(lmdb_path))
      val db: Database = env.openDatabase(null, 0)
      val part = thePart.asInstanceOf[LmdbPartition]
      val txn: Transaction = env.createReadTransaction()
      var pos_in_partition: Int = 0
      var it = if (txn != null)
        db.seek(txn, part.startKey)
      else {
        doClose(db)
        null
      }

      override def hasNext(): Boolean = {
        if (it == null) return false

        val res = it.hasNext
        if (!res || (pos_in_partition == part.size)) {
          doCommit(txn)
          doClose(db)
          it = null
          logInfo("Completed partition " + thePart.index)
        }
        res
      }

      override def next(): (String, String, Int, Int, Int, Boolean, Array[Byte]) = {
        if (it == null)
          ("", "", 0, 0, 0, false, null)
        else {
          val next = it.next()
          pos_in_partition = pos_in_partition + 1

          val id: String = new String(next.getKey())

          val datum_bdr = Datum.newBuilder()
          datum_bdr.mergeFrom(next.getValue())
          val datum = datum_bdr.build()

          val label: String = datum.getLabel().toString()
          //log.debug("ID:" + id + " label:" + label)

          val channels: Int = datum.getChannels()
          val height: Int = datum.getHeight()
          val width: Int = datum.getWidth()
          val encoded: Boolean = datum.getEncoded()
          val matData: Array[Byte] =
            if (encoded) datum.getData().toByteArray()
            else LmdbRDD.LMDBdata2Matdata(channels, height * width, datum.getData().toByteArray())

          (id, label, channels, height, width, encoded, matData)
        }
      }
    }

  private def doCommit(txn: Transaction): Unit = {
    try {
      if (txn != null) {
        txn.commit()
      }
    } catch {
      case e: Exception => logWarning("Exception commit transaction", e)
    }
  }

  private def doClose(db: Database): Unit = {
    try {
      if (db != null) {
        db.close()
      }
    } catch {
      case e: Exception => logWarning("Exception closing database", e)
    }
  }
}

private[caffe] object LmdbRDD {
  private val log: Logger = LoggerFactory.getLogger(this.getClass)
  private var libLoaded: Boolean = false
  private var fileMap: mutable.HashMap[String, String] = mutable.HashMap[String, String]()

  //load lmdbjni
  private def loadLibrary(): Unit = {
    synchronized {
      if (!libLoaded) {
        log.debug("java.library.path:" + System.getProperty("java.library.path"))
        System.loadLibrary("lmdbjni")
        log.debug("System load liblmdbjni.so successed")
        libLoaded = true
      }
    }
  }

  private def toLocalFile(sourceFilePath: String): String = {
    synchronized {
      if (fileMap.contains(sourceFilePath)) {
        fileMap.get(sourceFilePath).get
      } else {
        //download files if needed
        val local_lmdb_path =
          if (sourceFilePath.startsWith(FSUtils.localfsPrefix)) {
            sourceFilePath.substring("file://".length)
          } else {
            //copy .mdb files onto pwd
            val folder: Path = new Path(sourceFilePath)
            val fs: FileSystem = folder.getFileSystem(new Configuration())
            val pwd = System.getProperty("user.dir")
            val files_status_iterator = fs.listFiles(folder, false)
            while (files_status_iterator.hasNext) {
              val file_status = files_status_iterator.next()
              val src_path: Path = file_status.getPath()
              val src_fname = src_path.getName
              if (src_fname.endsWith(".mdb"))
                fs.copyToLocalFile(false, src_path, new Path("file://" + pwd + "/" + src_fname), true)
            }

            pwd
          }

        //make lmdb files wriatble
        mkWritable(local_lmdb_path)
        fileMap.put(sourceFilePath, local_lmdb_path)

        //return
        local_lmdb_path
      }
    }
  }

  private def mkWritable(lmdb_path: String): Unit = {
    //make sure that all mdb files are writable
    val db_files = new File(lmdb_path).listFiles(new FilenameFilter {
      override def accept(dir: File, name: String): Boolean =
        name.toLowerCase().endsWith(".mdb")
    })
    for (db_file <- db_files)
      db_file.setWritable(true)
  }

  /**
   * Pixel data reordered from LMDB format to cv:Mat format.
   * LMDB format is (channel, height, width), and data are (R, R,..., G, G, ..., B, B ...)
   * Mat format is (height, width, channel), and data are (R, G, B, R, G, B, ...)
   *
   * @param channels
   * @param dimension_size
   * @param data
   * @return
   */
  private[caffe] def LMDBdata2Matdata(channels: Int, dimension_size: Int, data: Array[Byte]): Array[Byte] = {
    if (channels == 1) data
    else {
      val data_clone = data.clone()

      for (p <- 0 until dimension_size)
        for (c <- 0 until channels)
          data_clone(p * channels + c) = data(p + c * dimension_size)

      data_clone
    }
  }
}
