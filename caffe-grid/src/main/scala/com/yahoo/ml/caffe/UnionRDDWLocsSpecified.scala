// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import org.apache.spark.{Partition, SparkContext}
import org.apache.spark.rdd.{RDD, UnionRDD}

import scala.reflect.ClassTag

private[caffe] class UnionRDDWLocsSpecified[T:ClassTag](sc: SparkContext, rdds: Seq[RDD[T]], locs: Array[String])
  extends UnionRDD[T](sc, rdds) {
  override def getPreferredLocations(s: Partition): Seq[String] = Seq(locs(s.index % locs.length))
}
