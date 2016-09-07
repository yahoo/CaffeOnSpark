// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import java.net.InetAddress
import org.apache.spark.{SparkContext, SparkEnv}
import org.apache.spark.rdd.RDD

private[caffe] object Util {
  def getSparkClassLoader(): ClassLoader =
    Option(Thread.currentThread().getContextClassLoader).getOrElse(getClass.getClassLoader)

  /**
   * Gather the task locations of our executors
   * @param sc SparkContext
   * @param size # of executors
   * @return Array of task locations
   */
  def executorLocations(sc: SparkContext, size: Int) : Array[String] = {
    val dummy_rdd : RDD[Int] = sc.parallelize(0 until size, size)
    dummy_rdd.mapPartitions{_ => {
      // Identify locations of executors with this prefix.
      val executorLocationTag = "executor_"
      //executor's hostname
      val host = InetAddress.getLocalHost.getHostName
      //executor ID
      val executorId = SparkEnv.get.executorId
      //location format per org/apache/spark/scheduler/TaskLocation.scala#L35
      Iterator(s"${executorLocationTag}${host}_$executorId")
    }}.collect()
  }
}
