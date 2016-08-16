package com.yahoo.ml.caffe

import org.apache.spark.Partitioner

class FixedSizePartitioner[V](partitions: Int, part_size: Int) extends Partitioner {
    def getPartition(key: Any): Int = {
      (key.asInstanceOf[Long] / part_size).toInt
    }

    def numPartitions: Int = partitions
}
