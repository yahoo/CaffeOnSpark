// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import java.io.{FilenameFilter, File, ObjectOutputStream, ByteArrayOutputStream}
import java.util.Calendar

import caffe.Caffe.Datum
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileStatus, FileSystem, Path}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.storage.StorageLevel
import org.fusesource.lmdbjni.Env
import org.slf4j.{LoggerFactory, Logger}

import scala.collection.mutable.ArrayBuffer


object LMDB {
  private val log: Logger = LoggerFactory.getLogger(this.getClass)
  private var libLoaded : Boolean = false

  def makeSequence(lmdb_path: String): Seq[(Array[Byte], Array[Byte])] = {
    //force lmdbjni.so to be loaded
    if (!libLoaded) {
      log.debug("java.library.path:"+System.getProperty("java.library.path"))
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

    //initialize sequence
    val seq = ArrayBuffer[(Array[Byte], Array[Byte])]()

    //iterate through LMDB and append entries into our sequence
    val env : Env = new Env(lmdb_path)
    val db = env.openDatabase(null, 0)
    val txn = env.createReadTransaction()
    try {
      val it = db.iterate(txn)
      while (it.hasNext) {
        val next = it.next()
        val id : String = new String(next.getKey())

        val datum_bdr = Datum.newBuilder()
        datum_bdr.mergeFrom(next.getValue())
        val datum = datum_bdr.build()

        val aout = new ByteArrayOutputStream
        val oos = new ObjectOutputStream(aout)
        val label : String = datum.getLabel().toString()
        oos.writeObject((id, label))
        //log.info("ID:" + key + " label:" + label)

        seq.append((aout.toByteArray(), datum.getData().toByteArray()))
        aout.close()
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
}

/**
 * LMDB is a built-in data source class for LMDB data source.
 * You could use this class for your LMDB data sources.
 *
 * @param conf CaffeSpark configuration
 * @param layerId the layer index in the network protocol file
 * @param isTrain
 */
class LMDB(conf: Config, layerId: Int, isTrain: Boolean) extends SeqImageDataSource(conf, layerId, isTrain) {
  /*
  TODO: We should revise the implementation to perform lazy fetching, instead of reads all LMDB entries into memory.
   */
  override def makeRDD(sc: SparkContext): RDD[(Array[Byte], Array[Byte])] = {
    val seq = if (sourceFilePath.startsWith(FSUtils.localfsPrefix)) {
      LMDB.makeSequence(sourceFilePath.substring("file://".length))
    } else {
      //copy .mdb files onto pwd
      val folder: Path = new Path(sourceFilePath)
      val fs: FileSystem = folder.getFileSystem(new Configuration())
      val pwd = System.getProperty("user.dir")
      val files_status_iterator  = fs.listFiles(folder, false)
      while (files_status_iterator.hasNext) {
        val file_status = files_status_iterator.next()
        val src_path: Path = file_status.getPath()
        val src_fname = src_path.getName
        if (src_fname.endsWith(".mdb"))
          fs.copyToLocalFile(false, src_path, new Path("file://" + pwd + "/" + src_fname), true)
      }

      LMDB.makeSequence(pwd)
    }

    if (seq.size == 0) {
      sc.emptyRDD[(Array[Byte], Array[Byte])]
    } else {
      sc.parallelize(seq).persist(StorageLevel.DISK_ONLY)
    }
  }

}
