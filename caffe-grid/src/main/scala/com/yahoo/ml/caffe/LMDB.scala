// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import java.io.{FilenameFilter, File}

import caffe.Caffe.Datum
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.Row
import org.fusesource.lmdbjni.Env
import org.slf4j.{LoggerFactory, Logger}

import scala.collection.mutable.ArrayBuffer


object LMDB {
  private val log: Logger = LoggerFactory.getLogger(this.getClass)
  private var libLoaded: Boolean = false

  private def prepare(lmdb_path: String): Unit = {
    //force lmdbjni.so to be loaded
    if (!libLoaded) {
      log.debug("java.library.path:" + System.getProperty("java.library.path"))
      System.loadLibrary("lmdbjni")
      log.debug("System load liblmdbjni.so successed")
      libLoaded = true
    }

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
  private def LMDBdata2Matdata(channels: Int, dimension_size: Int, data: Array[Byte]): Array[Byte] = {
    channels match {
      case 1 => data
      case 3 => {
        val data_clone = data.clone()
        for (i <- 0 until dimension_size) {
          data(i * 3) = data_clone(i)
          data(i * 3 + 1) = data_clone(i + dimension_size)
          data(i * 3 + 2) = data_clone(i + 2 * dimension_size)
        }
        data
      }
      case _ => {
        log.error("Unsupported # of channels")
        null
      }
    }
  }

  def makeSequence(lmdb_path: String): Seq[(String, String, Int, Int, Int, Boolean, Array[Byte])] = {
    //prepare LMDB
    prepare(lmdb_path)

    //initialize sequence
    val seq = ArrayBuffer[(String, String, Int, Int, Int, Boolean, Array[Byte])]()

    //iterate through LMDB and append entries into our sequence
    val env: Env = new Env(lmdb_path)
    val db = env.openDatabase(null, 0)
    val txn = env.createReadTransaction()
    try {
      val it = db.iterate(txn)
      while (it.hasNext) {
        val next = it.next()
        val id: String = new String(next.getKey())

        val datum_bdr = Datum.newBuilder()
        datum_bdr.mergeFrom(next.getValue())
        val datum = datum_bdr.build()

        val label: String = datum.getLabel().toString()
        //log.info("ID:" + id + " label:" + label)

        val channels: Int = datum.getChannels()
        val height: Int = datum.getHeight()
        val width: Int = datum.getWidth()
        val encoded: Boolean = datum.getEncoded()
        val matData = if (encoded) datum.getData().toByteArray()
                      else LMDBdata2Matdata(channels, height * width, datum.getData().toByteArray())
        seq.append((id, label, channels, height, width, encoded, matData))
      }
    } catch {
      case e: Exception =>
        log.warn(e.toString)
    } finally {
      txn.commit()
      db.close()
    }

    //return the sequence
    seq
  }

  def makeRowSeq(lmdb_path: String): Seq[Row] = {
    //prepare LMDB
    prepare(lmdb_path)

    //initialize sequence
    val list = ArrayBuffer[Row]()

    //iterate through LMDB and append entries into our sequence
    val env: Env = new Env(lmdb_path)
    val db = env.openDatabase(null, 0)
    val txn = env.createReadTransaction()
    try {
      val it = db.iterate(txn)
      while (it.hasNext) {
        val next = it.next()
        val id: String = new String(next.getKey())

        val datum_bdr = Datum.newBuilder()
        datum_bdr.mergeFrom(next.getValue())
        val datum = datum_bdr.build()

        val label: String = datum.getLabel().toString()
        //log.info("ID:" + id + " label:" + label)

        val channels: Int = datum.getChannels()
        val height: Int = datum.getHeight()
        val width: Int = datum.getWidth()
        val encoded: Boolean = datum.getEncoded()
        val matData = if (encoded) datum.getData().toByteArray()
                      else LMDBdata2Matdata(channels, height * width, datum.getData().toByteArray())
        list.append(Row(id, label, channels, height, width, encoded, matData))
      }
    } catch {
      case e: Exception =>
        log.warn(e.toString)
    } finally {
      txn.commit()
      db.close()
    }

    //return the list
    list
  }

  def toLocalFile(sourceFilePath: String): String = {
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
  }
}

/**
 * LMDB is a built-in data source class for LMDB data source.
 * You could use this class for your LMDB data sources.
 *
 * @param conf CaffeSpark configuration
 * @param layerId the layer index in the network protocol file
 * @param isTrain
 */
class LMDB(conf: Config, layerId: Int, isTrain: Boolean) extends ImageDataSource(conf, layerId, isTrain) {
  /*
  TODO: We should revise the implementation to perform lazy fetching, instead of reads all LMDB entries into memory.
   */
  override def makeRDD(sc: SparkContext): RDD[(String, String, Int, Int, Int, Boolean, Array[Byte])] = {
    val seq = LMDB.makeSequence(LMDB.toLocalFile(sourceFilePath))
    if (seq.size == 0) {
      sc.emptyRDD[(String, String, Int, Int, Int, Boolean, Array[Byte])]
    } else {
      sc.parallelize(seq, conf.clusterSize).persist(StorageLevel.DISK_ONLY)
    }
  }

}
