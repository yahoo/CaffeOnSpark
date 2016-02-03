// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import caffe.Caffe._
import com.yahoo.ml.jcaffe._

import java.io.FileReader
import java.nio.file.{StandardCopyOption, Files, Paths}

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.slf4j.{LoggerFactory, Logger}

private[caffe] object FSUtils {
  private[caffe] val localfsPrefix: String = "file:"
  private[caffe] val hdfsPrefix: String = "hdfs:"
  private[caffe] val log: Logger = LoggerFactory.getLogger(this.getClass)

  def CopyFileToHDFS(localFilePath: String, hdfsPath: String) {
    val dest: Path = new Path(hdfsPath)
    val fs: FileSystem = dest.getFileSystem(new Configuration())
    if (fs.exists(dest)) fs.delete(dest, true)
    fs.copyFromLocalFile(new Path("file://" + localFilePath), dest)
  }

  def CopyFileToLocal(hdfsPath: String, localFilePath: String) {
    val src: Path = new Path(hdfsPath)
    val fs: FileSystem = src.getFileSystem(new Configuration())
    val dest: Path = new Path("file://" + localFilePath)
    fs.copyToLocalFile(false, src, dest, true)
  }

  private def MatchH5Suffix(src: String, des: String): String = {
    var result = des

    if ((src.length > 3) && (result.length > 3)) {
      if (((src.substring(src.length - 3) == ".h5")) && (!(result.substring(result.length - 3) == ".h5")))
        result += ".h5"
      else if ((!(src.substring(src.length - 3) == ".h5")) && ((result.substring(result.length - 3) == ".h5")))
        result = result.substring(0, result.length - 3)
    }
    return result
  }

  private def CopyModelFile(caffeNet: CaffeNet, modelFilename: String, iterId: Int, isState: Boolean, useExactModelFilename: Boolean) {
    val localModelFilename: String = System.getProperty("user.dir") + "/" + caffeNet.snapshotFilename(iterId, isState)

    var desModelFilename = modelFilename
    if (!useExactModelFilename) {
      val pathidx: Int = modelFilename.lastIndexOf("/")
      val fnidx: Int = localModelFilename.lastIndexOf("/")
      desModelFilename = desModelFilename.substring(0, pathidx + 1) + localModelFilename.substring(fnidx + 1)
    }

    desModelFilename = MatchH5Suffix(localModelFilename, desModelFilename)
    log.info("destination file:"+desModelFilename)
    if (modelFilename.startsWith(localfsPrefix)) {
      desModelFilename = desModelFilename.substring(localfsPrefix.length)
      val srcPath: java.nio.file.Path = Paths.get(localModelFilename)
      val desPath: java.nio.file.Path = Paths.get(desModelFilename)
      log.info(srcPath+"-->"+desPath)
      Files.move(srcPath, desPath, StandardCopyOption.REPLACE_EXISTING)
    }
    else
      CopyFileToHDFS(localModelFilename, desModelFilename)
  }

  def GenModelOrState(caffeNet: CaffeNet, modelFilename: String, genState: Boolean) {
    val iterID = caffeNet.snapshot()
    CopyModelFile(caffeNet, modelFilename, iterID, false, !genState)
    if (genState)
      CopyModelFile(caffeNet, modelFilename, iterID, true, false)
  }

  def GetLocalFileName(fileName: String, tmpFileName: String): String = {
    var localFileName = ""
    if (fileName.startsWith(localfsPrefix))
      localFileName = fileName.substring(localfsPrefix.length)
    else if (fileName.length > 3) {
      localFileName = System.getProperty("user.dir") + "/" + tmpFileName
      if (fileName.substring(fileName.length - 3) == ".h5")
        localFileName = localFileName + ".h5"
      CopyFileToLocal(fileName, localFileName)
    }

    return localFileName
  }
}
