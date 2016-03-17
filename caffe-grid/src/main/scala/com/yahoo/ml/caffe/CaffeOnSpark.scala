// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import java.io.{FileReader, PrintWriter}
import java.net.InetAddress

import caffe.Caffe._

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql
import org.apache.spark.sql.types.{FloatType, StructField, StructType, ArrayType, StringType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.functions.udf

import org.slf4j.{LoggerFactory, Logger}

import scala.collection.mutable

object CaffeOnSpark {
  private val log: Logger = LoggerFactory.getLogger(this.getClass)

  def main(args: Array[String]) {
    val sc_conf = new SparkConf()
    sc_conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.scheduler.minRegisteredResourcesRatio", "1.0")
    val sc: SparkContext = new SparkContext(sc_conf)

    //Caffe-on-Spark configuration
    var conf = new Config(sc, args)
    
    //training if specified
    val caffeSpark = new CaffeOnSpark(sc)
    if (conf.isTraining) {
      val source = DataSource.getSource(conf, true)
      caffeSpark.train(source)
    }

    //feature extraction
    if (conf.isFeature || conf.isTest) {
      val source = DataSource.getSource(conf, false)
      if (conf.isFeature) { //feature extraction
        val featureDF = caffeSpark.features(source)
        
        //save extracted features into the specified file
        val rdf = featureDF.write.format(source.conf.outputFormat).save(source.conf.outputPath)
      }else { //test
        val result = caffeSpark.test(source)

        //save test results into a local file
        val outputPath = source.conf.outputPath
        var localFilePath: String = outputPath
        if (outputPath.startsWith(FSUtils.localfsPrefix))
          localFilePath = outputPath.substring(FSUtils.localfsPrefix.length)
        else
          localFilePath = System.getProperty("user.dir") + "/test_result.tmp"
        val out: PrintWriter = new PrintWriter(localFilePath)
        result.map{
          case (name, r) => {
            out.println(name + ": "+r.mkString(","))
          }
        }
        out.close

        //upload the result file available on HDFS
        if (!outputPath.startsWith(FSUtils.localfsPrefix))
          FSUtils.CopyFileToHDFS(localFilePath, outputPath)
      }

    }
  }
}



/**
 * CaffeOnSpark is the main class for distributed deep learning.
 * It will launch multiple Caffe cores within Spark executors, and conduct coordinated learning from HDFS datasets.
 *
 * @param sc Spark Context
 */
class CaffeOnSpark(@transient val sc: SparkContext) extends Serializable {
  @transient private val log: Logger = LoggerFactory.getLogger(this.getClass)
  @transient val floatarray2doubleUDF = udf((float_features: Seq[Float]) => {
    float_features(0).toDouble
  })
  @transient val floatarray2doublevectorUDF = udf((float_features: Seq[Float]) => {
    val double_features = new Array[Double](float_features.length)
    for (i <- 0 until float_features.length) double_features(i) = float_features(i)
    Vectors.dense(double_features)
  })

  /**
   * Training with a specific data source
   * @param source input data source
   */
  def train[T1,T2](source:DataSource[T1,T2]): Unit = {
    var trainDataRDD : RDD[T1] = source.makeRDD(sc)
    if (trainDataRDD == null) {
      log.info("No training data is given")
      return
    }

    //Phase 1: Gather RDMA addresses from executors
    val conf = source.conf
    if (!conf.snapshotStateFile.isEmpty && conf.snapshotModelFile.isEmpty) {
      log.error("to resume training, please provide input model file")
      return
    }

    var rank_2_addresses_n_host = sc.parallelize(0 until conf.clusterSize, conf.clusterSize).map {
      case rank: Int => {
        val processor = CaffeProcessor.instance[T1, T2](source, rank)

        //announce local RDMA address
        if (conf.clusterSize > 1) {
          (rank, processor.getLocalAddress(), InetAddress.getLocalHost.getHostName)
        } else {
          (rank, new Array[String](1), InetAddress.getLocalHost.getHostName)
        }
      }
    }.collect()

    for (i <- rank_2_addresses_n_host)
      log.info("rank = " + i._1 + ", RDMAaddress = " + i._2.mkString(",") + ", hostname = " + i._3)
    var numExecutors: Int = sc.getExecutorMemoryStatus.size
    val numDriver: Int = if (sc.isLocal) 0 else 1
    if (conf.clusterSize + numDriver != numExecutors) {
      log.error("Requested # of executors: " + conf.clusterSize + " actual # of executors:" + (numExecutors - numDriver) +
      ". Please try to set --conf spark.scheduler.maxRegisteredResourcesWaitingTime with a large value (default 30s)")
      throw new IllegalStateException("actual number of executors is not as expected")
    }

    //Phase 2: bcast RDMA addresses
    val rank_2_addresses = rank_2_addresses_n_host.map {
      case (rank, rdma_addr, host) => {
        if (rank==0) log.info("rank 0:"+host)
        (rank, rdma_addr)
      }
    }
    val bcast_addresses = sc.broadcast(rank_2_addresses)

    //Phase 3: set up the processors
    sc.parallelize(0 until conf.clusterSize, conf.clusterSize).map {
      case rank: Int => {
        val processor = CaffeProcessor.instance[T1,T2]()
        //start processor w/ the given addresses
        processor.start(bcast_addresses.value)
      }
    }.collect()

    //Phase 4: repartition RDD if needed
    val origin_part_count = trainDataRDD.partitions.size
    val desired_part_count = (origin_part_count / conf.clusterSize) * conf.clusterSize
    if (origin_part_count != desired_part_count) {
      trainDataRDD = trainDataRDD.coalesce(desired_part_count, true)
      log.info("Training dataset partition count: "+origin_part_count+" -> "+desired_part_count)
    }
    if (conf.isRddPersistent) {
      trainDataRDD = trainDataRDD.persist(StorageLevel.DISK_ONLY)
    }

    //Phase 5: find the minimum size of partitions
    var minPartSize = 0
    if (conf.clusterSize > 1) {
      val sizeRDD = trainDataRDD.mapPartitions {
        iter => {
          val partSize = iter.size
          // synchronize among the executors,
          // to achieve same number of partitions.
          val processor = CaffeProcessor.instance[T1,T2]()
          processor.sync()
          Iterator(partSize)
        }
      }.persist()
      minPartSize = sizeRDD.min()
      log.info("Partition size: min=" + minPartSize + " max="+sizeRDD.max())
    }

    //Phase 6: feed the processor
    var continue: Boolean = true
    while (continue) {
      //conduct training with dataRDD
      continue = trainDataRDD.mapPartitions{
        iter => {
          var res = false
          //feed training data from iterator
          val processor = CaffeProcessor.instance[T1,T2]()
          if (!processor.solversFinished) {
            if (minPartSize > 0) {
              res = iter.take(minPartSize).map { sample => processor.feedQueue(sample) }.reduce(_ && _)
            } else {
              res = iter.map { sample => processor.feedQueue(sample) }.reduce(_ && _)
            }
            processor.solversFinished = !res
          }
          Iterator(res)
        }
      }.reduce(_ && _)
    }

    //Phase 7: shutdown processors
    shutdownProcessors(conf)
  }

  /**
   * a utility function for shutting processor thread pool
   */
  private def shutdownProcessors[T1,T2](conf: Config): Unit = {
    sc.parallelize(0 until conf.clusterSize, conf.clusterSize).map {
      _ => {
        val processor = CaffeProcessor.instance[T1, T2]()
        processor.stop()
      }
    }.collect()
  }

  /**
   * Test with a specific data source.
   * Test result will be saved into HDFS file per configuration.
   *
   * @param source input data source
   * @return key/value map for mean values of output layers
   */
  def test[T1,T2](source:DataSource[T1,T2]) : Map[String, Seq[Double]] = {
    val testDF = features2(source)

    var result = new mutable.HashMap[String, Seq[Double]]
    // compute the mean of the columns
    testDF.columns.zipWithIndex.map {
      case (name, index) => {
        if (index > 0) { // first column is SampleId, ignored.
          val n: Int = testDF.take(1)(0).getSeq[Double](index).size
          val ndf = testDF.agg(new VectorMean(n)(testDF(name)))
          val r: Seq[Double] = ndf.take(1)(0).getSeq[Double](0)
          result(name) = r
        }
      }
    }

    //shutdown processors
    shutdownProcessors(source.conf)

    result.toMap
  }

  /**
   * Extract features from a specific data source.
   * Features will be saved into DataFrame per configuration.
   *
   * @param source input data source
   * @return Feature data frame
   */
  def features[T1,T2](source:DataSource[T1,T2]) : DataFrame = {
    var featureDF = features2(source)

    //take action to force featureDF persisted
    featureDF.count()

    //shutdown processors
    shutdownProcessors(source.conf)

    featureDF
  }

  /**
   * Extract features from a data source
   * @param source input data source
   * @return a data frame
   */
  private def features2[T1,T2](source:DataSource[T1,T2]) : DataFrame = {
    val srcDataRDD = source.makeRDD(sc)
    val conf = source.conf
    val clusterSize: Int = conf.clusterSize

    //Phase 1: start Caffe processor within each executor
    val size = sc.parallelize(0 until clusterSize, clusterSize).map {
      case rank: Int => {
        // each processor has clusterSize 1 and rank 0
        val processor = CaffeProcessor.instance[T1,T2](source, rank)
      }
    }.count()
    if (size < clusterSize) {
      log.error((clusterSize-size) + "executors have failed. Please check Spark executor logs")
      throw new IllegalStateException("Executor failed at CaffeProcessor startup for test/feature extraction")
    }

    // Sanity check
    val numExecutors: Int = sc.getExecutorMemoryStatus.size
    val numDriver: Int = if (sc.isLocal) 0 else 1
    if ((size + numDriver) != sc.getExecutorMemoryStatus.size) {
      log.error("Requested # of executors: " + clusterSize + " actual # of executors:" + (numExecutors-numDriver) +
        ". Please try to set --conf spark.scheduler.maxRegisteredResourcesWaitingTime with a large value (default 30s)")
      throw new IllegalStateException("actual number of executors is not as expected")
    }

    // Phase 2 get output schema
    val blobNames = if (conf.isFeature)
      conf.features
    else // this is test mode
      sc.parallelize(0 until clusterSize, clusterSize).map { _ =>
        val processor = CaffeProcessor.instance[T1, T2]()
        processor.getTestOutputBlobNames
      }.collect()(0)
    val schema = new StructType(Array(StructField("SampleID", StringType, false)) ++ blobNames.map(name => StructField(name, ArrayType(FloatType), false)))
    log.info("Schema:" + schema)

    //Phase 3: feed the processors
    val featureRDD = srcDataRDD.mapPartitions {
      iter => {
        val processor = CaffeProcessor.instance[T1,T2]()
        if (!processor.solversFinished) {
          processor.start(null)
          val res = iter.map { sample => processor.feedQueue(sample) }.reduce(_ && _)
          processor.solversFinished = !res
          processor.stopThreads()
          import scala.collection.JavaConversions._
          processor.results.iterator
        } else {
          Iterator()
        }
      }
    }

    //Phase 4: Create output data frame
    val sqlContext = new sql.SQLContext(sc)
    sqlContext.createDataFrame(featureRDD, schema).persist(StorageLevel.DISK_ONLY)
  }

}


