// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import org.apache.spark.Partitioner

private class FixedSizePartitioner[V](partitions: Int, part_size: Int) extends Partitioner {
    def getPartition(key: Any): Int = {
      (key.asInstanceOf[Long] / part_size).toInt
    }

    def numPartitions: Int = partitions
}
