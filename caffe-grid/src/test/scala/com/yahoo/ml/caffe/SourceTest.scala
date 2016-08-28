// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import java.io.File
import com.yahoo.ml.caffe.tools.Binary2Sequence
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.slf4j.LoggerFactory
import org.testng.Assert._

import caffe.Caffe._
import com.yahoo.ml.jcaffe._

class SourceTest extends FunSuite with BeforeAndAfterAll {
  val log = LoggerFactory.getLogger(this.getClass)
  var sc : SparkContext = null
  var conf : Config = null

  override def beforeAll() = {
    sc = new SparkContext(new SparkConf().setAppName("caffe-on-spark").setMaster("local"))

    val ROOT_PATH = {
      val fullPath = getClass.getClassLoader.getResource("log4j.properties").getPath
      fullPath.substring(0, fullPath.indexOf("caffe-grid/"))
    }
    val solver_config_path = ROOT_PATH + "caffe-grid/src/test/resources/caffenet_solver.prototxt";
    val args = Array("-conf", solver_config_path,
      "-model", "file:"+ROOT_PATH+"caffe-grid/target/model.h5",
      "-imageRoot", "file:"+ROOT_PATH+"data/images",
      "-labelFile", "file:"+ROOT_PATH+"data/images/labels.txt"
    )
    conf = new Config(sc, args)

    val seq_file_path = "file:"+ROOT_PATH+"caffe-grid/target/seq_image_files"
    val path = new Path(seq_file_path)
    val fs = path.getFileSystem(new Configuration)
    if (fs.exists(path)) fs.delete(path, true)

    val b2s = new Binary2Sequence(sc, conf)
    assertNotNull(b2s)
    b2s.makeRDD().saveAsSequenceFile(seq_file_path)
  }

  override def afterAll() = {
    sc.stop()
  }

  test("Config test") {
    assert(conf.clusterSize >= 1)
    assertEquals(conf.train_data_layer_id, 0)
    assertEquals(conf.test_data_layer_id, 1)
  }

  test("Training source") {
    val source =  new SeqImageDataSource(conf, conf.train_data_layer_id, true)

    //caffenet
    val net = new CaffeNet(conf.protoFile, "", "", 1, 1, 0, true, 0, -1, 0)
    assertTrue(net != null)
    assertTrue(net.connect(null))

    //initialization
    assertTrue(source.init())
    assertTrue(source.batchSize() > 0)
    assertTrue(source.isTrain)
    source.resetQueue()

    //RDD
    val rdd = source.makeRDD(sc).persist()
    assertNotNull(rdd)

    //dummy objects
    val batchSize = source.batchSize()
    val data = source.dummyDataBlobs()
    assertNotNull(data)
    val dataholder = source.dummyDataHolder()
    val matVector = dataholder._1
    val label = dataholder._2
    assertNotNull(matVector)
    assertNotNull(label)
    val sampleIds = new Array[String](batchSize)

    //source transformer
    val transformer = new FloatDataTransformer(source.transformationParameter, true)
    assertNotNull(transformer)

    //simplified training
    log.info("SourceTest train:")
    for (i <- 0 until 5) {
        //feed training data to source queue
        var res = rdd.take(source.batchSize()).map(source.offer(_)).reduce(_ && _)
        assertTrue(res)

        //next batch
        res = source.nextBatch(sampleIds, dataholder)
        assertTrue(res)

        //tranform the data blob
        transformer.transform(matVector, data(0))

        //copy label
        data(1).copyFrom(label)

        //training
        assertTrue(net.train(0, data))
    }

    //save snapshot
    val filePath = conf.modelPath.substring("file:".length);
    val f = new File(filePath)
    if (f.exists()) f.delete()
    FSUtils.GenModelOrState(net, conf.modelPath, false)
    assertTrue(new File(filePath).exists())

    net.deallocate();
  }

  test("Test source") {
    val source =  new SeqImageDataSource(conf, conf.test_data_layer_id, false)

    //caffenet
    val test_net = new CaffeNet(conf.protoFile, "", "", 1, 1, 0, false, 0, -1, 0)
    assertTrue(test_net != null)
    assertTrue(test_net.connect(null))

    //initialization
    assertTrue(source.init())
    assertTrue(source.batchSize() > 0)
    assertTrue(!source.isTrain)
    source.resetQueue()

    //RDD
    val rdd = source.makeRDD(sc)
    assertNotNull(rdd)


    //dummy objects
    val batchSize = source.batchSize()
    val data = source.dummyDataBlobs()
    assertNotNull(data)
    val dataholder = source.dummyDataHolder()
    val matVector = dataholder._1
    val label = dataholder._2
    assertNotNull(matVector)
    assertNotNull(label)
    val sampleIds = new Array[String](batchSize)

    //source transformer
    val transformer = new FloatDataTransformer(source.transformationParameter, false)
    assertNotNull(transformer)

    //feed training data to source queue
    var res = rdd.take(source.batchSize()).map(source.offer(_)).reduce(_ && _)
    assertTrue(res)

    //next batch
    res = source.nextBatch(sampleIds, dataholder)
    assertTrue(res)

    //tranform the data blob
    transformer.transform(matVector, data(0))

    //copy label
    data(1).copyFrom(label)

    //test
    val test_features : Array[String] = Array("accuracy","loss")
    val top_blobs_vec = test_net.predict(0, data, test_features)

    //validate test results
    for (j <- 0 until top_blobs_vec.length) {
      val result_vec = top_blobs_vec(j).cpu_data
      assertTrue(result_vec.get(0) < 50.0)
    }

    test_net.deallocate()
  }
}
