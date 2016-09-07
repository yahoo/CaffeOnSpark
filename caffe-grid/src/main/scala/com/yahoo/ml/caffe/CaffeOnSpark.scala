// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import java.io.PrintWriter
import java.net.InetAddress

import caffe.Caffe._

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkEnv, SparkContext, SparkConf, sql}
import org.apache.spark.sql.types.{FloatType, StructField, StructType, ArrayType, StringType}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.functions.udf

import org.slf4j.{LoggerFactory, Logger}
import scala.collection.mutable
import scala.collection.immutable.Map
import org.apache.spark.rdd._
import scala.reflect.ClassTag

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
    if (conf.isTraining ){
      if (conf.solverParameter.hasTestInterval && (conf.solverParameter.getTestInterval() != 0) && (conf.solverParameter.getTestIter(0) != 0)) {
        val sourceTrain: DataSource[Any,Any] = DataSource.getSource(conf, true).asInstanceOf[DataSource[Any, Any]]
        val sourceValidation: DataSource[Any,Any] = DataSource.getSource(conf, false).asInstanceOf[DataSource[Any, Any]]
        caffeSpark.trainWithValidation(sourceTrain, sourceValidation)
      } else {
        val sourceTrain: DataSource[Any,Any] = DataSource.getSource(conf, true).asInstanceOf[DataSource[Any, Any]]
        caffeSpark.train(sourceTrain)
      }
    }

    //feature extraction
    if (conf.isFeature || conf.isTest) {
      val source = DataSource.getSource(conf, false)
      if (conf.isFeature) {
        //feature extraction
        val featureDF = caffeSpark.features(source)

        //save extracted features into the specified file
        val rdf = featureDF.write.format(source.conf.outputFormat).save(source.conf.outputPath)
      } else {
        //test
        val result = caffeSpark.test(source)

        //save test results into a local file
        val outputPath = source.conf.outputPath
        var localFilePath: String = outputPath
        if (outputPath.startsWith(FSUtils.localfsPrefix))
          localFilePath = outputPath.substring(FSUtils.localfsPrefix.length)
        else
          localFilePath = System.getProperty("user.dir") + "/test_result.tmp"
        val out: PrintWriter = new PrintWriter(localFilePath)
        result.map {
          case (name, r) => {
            out.println(name + ": " + r.mkString(","))
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
  @transient val sqlContext = new sql.SQLContext(sc)
  @transient val floatarray2doubleUDF = udf((float_features: Seq[Float]) => {
    float_features(0).toDouble
  })
  @transient val floatarray2doublevectorUDF = udf((float_features: Seq[Float]) => {
    val double_features = new Array[Double](float_features.length)
    for (i <- 0 until float_features.length) double_features(i) = float_features(i)
    Vectors.dense(double_features)
  })

  private def setupTraining[T1, T2](sources: Array[DataSource[T1, T2]]): Array[String] = {
    //Phase 1: Gather RDMA addresses from executors
    val conf = sources(0).conf
    if (!conf.snapshotStateFile.isEmpty && conf.snapshotModelFile.isEmpty) {
      log.error("to resume training, please provide input model file")
      throw new IllegalStateException("input model file must be provided for incremental training")
    }

    var rank_2_addresses_n_host = sc.parallelize(0 until conf.clusterSize, conf.clusterSize).map {
      case rank: Int => {
        val processor = CaffeProcessor.instance[T1, T2](sources, rank)
        //announce local RDMA address
        if (conf.clusterSize > 1) {
          (rank, processor.getLocalAddress(), InetAddress.getLocalHost.getHostName)
        } else {
          (rank, new Array[String](1), InetAddress.getLocalHost.getHostName)
        }
      }
    }.collect()

    for (i <- rank_2_addresses_n_host)
      log.info("rank = " + i._1 + ", address = " + i._2.mkString(",") + ", hostname = " + i._3)
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
        if (rank == 0) log.info("rank 0:" + host)
        (rank, rdma_addr)
      }
    }
    val bcast_addresses = sc.broadcast(rank_2_addresses)

    //Phase 3: set up the processors
    val validation_blob_names = sc.parallelize(0 until conf.clusterSize, conf.clusterSize).map {
      case rank: Int => {
        val processor = CaffeProcessor.instance[T1, T2]()
        //start processor w/ the given addresses
        processor.start(bcast_addresses.value)

        if (rank==0) processor.getValidationOutputBlobNames()
        else null
      }
    }.collect()

    //return validation blob names if any
    if (validation_blob_names.length>=1) validation_blob_names.apply(0) else null
  }

  /**
   * Training with a specific data source
   * @param source input data source
   */
  def train[T1, T2](source: DataSource[T1, T2]): Unit = {
    var trainDataRDD: RDD[T1] = source.makeRDD(sc)
    if (trainDataRDD == null) {
      log.info("No training data is given")
      throw new IllegalStateException("No training data is given")
    }

    setupTraining(Array(source))
    val conf = source.conf
    //Phase 1: repartition RDD if needed
    val origin_part_count = trainDataRDD.partitions.size
    val desired_part_count = (origin_part_count / conf.clusterSize) * conf.clusterSize
    if (origin_part_count != desired_part_count) {
      trainDataRDD = trainDataRDD.coalesce(desired_part_count, true)
      log.info("Training dataset partition count: " + origin_part_count + " -> " + desired_part_count)
    }
    if (conf.isRddPersistent) {
      trainDataRDD = trainDataRDD.persist(StorageLevel.DISK_ONLY)
    }

    //Phase 2: find the minimum size of partitions
    var minPartSize = 0
    if (conf.clusterSize > 1) {
      val sizeRDD = trainDataRDD.mapPartitions {
        iter => {
          val partSize = iter.size
          // Spark decides how data partitions are distributed among executors in this step.
          // synchronize among the executors,
          // to achieve same number of partitions.
          val processor = CaffeProcessor.instance[T1, T2]()
          processor.sync()
          Iterator(partSize)
        }
      }.persist()
      minPartSize = sizeRDD.min()
      log.info("Partition size: min=" + minPartSize + " max=" + sizeRDD.max())
    }

    //Phase 3: feed the processor    
    var continuetrain: Boolean = true
    while (continuetrain) {
      	//conduct training with dataRDD
      	continuetrain = trainDataRDD.mapPartitions {
       	  iter => {
            var res = false
            //feed training data from iterator
            val processor = CaffeProcessor.instance[T1, T2]()
            if (!processor.solversFinished) {
              if (minPartSize > 0) {
                var idx = 0
                //the entire iterator needs to be consumed, otherwise GC won't be triggered
                res = iter.map { sample => {
                  idx += 1
                  if (idx <= minPartSize) processor.feedQueue(0, sample) else true
                }}.reduce(_ && _)
              } else {
                res = iter.map { sample => processor.feedQueue(0, sample) }.reduce(_ && _)
              }
              processor.solversFinished = !res
            }
            Iterator(res)
          }
        }.reduce(_ && _)
      }
    
    //Phase 4: shutdown processors
    shutdownProcessors(conf)
  }

  /**
   * Training interleaved with validation
   * @param sourceTrain input data source for training
   * @param sourceValidation input data source for validation
   * @return DataFrame of validation results
   */
  def trainWithValidation[T1, T2](sourceTrain: DataSource[T1, T2], sourceValidation: DataSource[T1, T2]): DataFrame = {
    log.info("interleave")
    var trainDataRDD: RDD[T1] = sourceTrain.makeRDD(sc)
    if (trainDataRDD == null) {
      log.info("No training data given")
      throw new IllegalStateException("No training data given")
    }

    var validationDataRDD: RDD[T1] = sourceValidation.makeRDD(sc)
    if (validationDataRDD == null) {
      log.info("No validation data given")
      throw new IllegalStateException("No validation data given")
    }

    val conf = sourceTrain.conf
    //Create train and test RDDs from parent RDD
    var continue: Boolean = true
    val no_of_records_required_per_partition_train = conf.solverParameter.getTestInterval() * sourceTrain.batchSize()  * conf.devices
    val total_records_train = trainDataRDD.count()
    log.info("total_records_train: " + total_records_train)
    log.info("no_of_records_required_per_partition_train: " + no_of_records_required_per_partition_train)
    if (total_records_train < no_of_records_required_per_partition_train * conf.clusterSize) {
      throw new IllegalStateException("Insufficient training data. Please adjust hyperparameters or increase dataset.")
    }
    val no_of_partitions_train = (total_records_train/no_of_records_required_per_partition_train).toInt
    log.info("num of training partitions: " + no_of_partitions_train)

    val num_records_per_validation_partition = conf.solverParameter.getTestIter(0) * sourceValidation.batchSize()
    val total_records_validation = validationDataRDD.count()
    log.info("total_records_validation: " + total_records_validation)
    log.info("num_records_per_validation_partition: " + num_records_per_validation_partition)
    if (total_records_validation < num_records_per_validation_partition) {
      throw new IllegalStateException("Insufficient validation data. Please adjust hyperparameters or increase dataset.")
    }
    val num_validation_parts = (total_records_validation/num_records_per_validation_partition).toInt
    log.info("num of validation partitions: " + num_validation_parts)

    val validationOutputBlobNames = setupTraining(Array(sourceTrain, sourceValidation))

    implicit val rdd_class_tag : ClassTag[T1] = ClassTag.apply[T1](trainDataRDD.first.getClass)
    val repartitionedTrainRDD = partitionRddWithFixedSize(trainDataRDD,
      no_of_records_required_per_partition_train, no_of_partitions_train, conf.isRddPersistent)
    val repartitionedValidationRDD = partitionRddWithFixedSize(validationDataRDD,
      num_records_per_validation_partition, num_validation_parts, false)

    //interleaved training RDDs
    val num_train_iters = no_of_partitions_train/conf.clusterSize
    val interleaveTrainRDDs:Array[RDD[(Long,T1)]] = new Array[RDD[(Long,T1)]](num_train_iters)
    for (i <- 0 until num_train_iters)
      interleaveTrainRDDs(i) = PartitionPruningRDD.create(repartitionedTrainRDD,
        (index => (index >= i*conf.clusterSize) && (index < (i+1)*conf.clusterSize)))


    //interleaved validation RDDs
    val interleaveValidationRDDs:Array[RDD[(Long,T1)]] = new Array[RDD[(Long,T1)]](num_validation_parts)
    val executorLocs = Util.executorLocations(sc, conf.clusterSize)
    for (i <- 0 until num_validation_parts) {
      //Create the interleaveValidationRDD for the required range
      interleaveValidationRDDs(i) = PartitionPruningRDD.create(repartitionedValidationRDD, (_ == i))
      if (conf.clusterSize>1)
        interleaveValidationRDDs(i) = new UnionRDDWLocsSpecified(sc,
          Array.fill(conf.clusterSize)(interleaveValidationRDDs(i)),
          executorLocs)
    }

    var current_train_iter = 0
    var current_validation_iter = 0
    while(continue) {
      //Proceed with the training
      continue = interleaveTrainRDDs(current_train_iter).mapPartitions {
        iter => {
          var res = false
          //feed training data from iterator
          val processor = CaffeProcessor.instance[T1, T2]()
          if (!processor.solversFinished) {
            processor.sync()
            res = iter.map { sample => processor.feedQueue(0, sample._2) }.reduce(_ && _)
            processor.solversFinished = !res
          }
          Iterator(res)
        }
      }.reduce(_ && _)

      if (continue) {
        //Proceed with the validation
        interleaveValidationRDDs(current_validation_iter).mapPartitions {
          iter => {
            //feed validation data from iterator
            val processor = CaffeProcessor.instance[T1, T2]()
            if (!processor.solversFinished) {
              processor.sync()
              val res = iter.map { sample => processor.feedQueue(1, sample._2)}.reduce(_ && _)
              processor.solversFinished = !res
            }

            Iterator(1)
          }
        }.collect()

        current_train_iter = (current_train_iter + 1) % num_train_iters
        current_validation_iter = (current_validation_iter + 1) % num_validation_parts
      }
    }

    //collect validation result from one processor
    val validation_output: Array[Row] = sc.parallelize(1 to 1, 1).mapPartitions{
      _ => {
          val processor = CaffeProcessor.instance[T1, T2]()
          processor.validationResults.toIterator
      }
    }.collect()

    //shutdown processors
    shutdownProcessors(conf)

    //dataframe of validation result
    import scala.collection.JavaConverters._
    val schema = new StructType(validationOutputBlobNames.map(name => StructField(name, ArrayType(FloatType), false)))
    sqlContext.createDataFrame(validation_output.toList.asJava, schema).persist(StorageLevel.DISK_ONLY)
  }

  /*
  construct a new RDD with fixed size partitions from a given RDD.
   */
  private def partitionRddWithFixedSize[T1:ClassTag](rdd:RDD[T1],
                 part_len: Int, num_parts: Int, persistent:Boolean): RDD[(Long,T1)] = {
    val partitioner = new FixedSizePartitioner(num_parts+1, part_len)
    var partitioned_rdd = rdd.zipWithIndex.map{case (e,i) => (i,e)}.partitionBy(partitioner)

    if (persistent) {
      partitioned_rdd = partitioned_rdd.persist(StorageLevel.DISK_ONLY)
      //unpersist the original RDD
      rdd.unpersist()
    }

    partitioned_rdd
  }

  /**
   * a utility function for shutting processor thread pool
   */
  private def shutdownProcessors[T1, T2](conf: Config): Unit = {
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
  def test[T1, T2](source: DataSource[T1, T2]): Map[String, Seq[Double]] = {
    source.conf.isTest = true
    val testDF = features2(source)

    var result = new mutable.HashMap[String, Seq[Double]]
    // compute the mean of the columns
    testDF.columns.zipWithIndex.map {
      case (name, index) => {
        if (index > 0) {
          // first column is SampleId, ignored.
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
  def features[T1, T2](source: DataSource[T1, T2]): DataFrame = {
    source.conf.isTest = false
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
  private def features2[T1, T2](source: DataSource[T1, T2]): DataFrame = {
    val srcDataRDD = source.makeRDD(sc)
    val conf = source.conf
    val clusterSize: Int = conf.clusterSize

    //Phase 1: start Caffe processor within each executor
    val size = sc.parallelize(0 until clusterSize, clusterSize).map {
      case rank: Int => {
        // each processor has clusterSize 1 and rank 0
        val processor = CaffeProcessor.instance[T1, T2](Array(source), rank)
      }
    }.count()
    if (size < clusterSize) {
      log.error((clusterSize - size) + "executors have failed. Please check Spark executor logs")
      throw new IllegalStateException("Executor failed at CaffeProcessor startup for test/feature extraction")
    }

    // Sanity check
    val numExecutors: Int = sc.getExecutorMemoryStatus.size
    val numDriver: Int = if (sc.isLocal) 0 else 1
    if ((size + numDriver) != sc.getExecutorMemoryStatus.size) {
      log.error("Requested # of executors: " + clusterSize + " actual # of executors:" + (numExecutors - numDriver) +
        ". Please try to set --conf spark.scheduler.maxRegisteredResourcesWaitingTime with a large value (default 30s)")
      throw new IllegalStateException("actual number of executors is not as expected")
    }

    // Phase 2 get output schema
    val blobNames = if (conf.isFeature)
      conf.features
    else // this is test mode
      sc.parallelize(0 until clusterSize, clusterSize).map { _ =>
        val processor = CaffeProcessor.instance[T1, T2]()
        processor.getValidationOutputBlobNames()
      }.collect()(0)
    val schema = new StructType(Array(StructField("SampleID", StringType, false)) ++ blobNames.map(name => StructField(name, ArrayType(FloatType), false)))
    log.info("Schema:" + schema)

    //Phase 3: feed the processors
    val featureRDD = srcDataRDD.mapPartitions {
      iter => {
        val processor: CaffeProcessor[T1, T2] = CaffeProcessor.instance[T1, T2]()
        val feature_iter: Iterator[Row] =
          if (processor.solversFinished)
            Iterator()
          else {
            processor.synchronized {
              processor.start(null)
              val res = iter.map { sample => processor.feedQueue(0, sample) }.reduce(_ && _)
              processor.solversFinished = !res
              processor.stopThreads()

              import scala.collection.JavaConversions._
              processor.results.iterator
            }
          }
        feature_iter
      }
    }

    //Phase 4: Create output data frame
    sqlContext.createDataFrame(featureRDD, schema).persist(StorageLevel.DISK_ONLY)
  }

}


