package com.yahoo.ml.caffe

import org.apache.spark.Partitioner

private class FixedSizePartitioner[V](partitions: Int, part_size: Int, dups: Int) extends Partitioner {
    def getPartition(key: Any): Int = {
      val key_long = key.asInstanceOf[Long]
      if (dups<=1)
        (key_long / part_size).toInt
      else
        (key_long / (part_size*dups)).toInt * dups + (key_long % dups).toInt
    }

    def numPartitions: Int = partitions
}
