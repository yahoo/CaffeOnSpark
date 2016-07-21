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
import scala.math.ceil

class InterleaveTest extends FunSuite with BeforeAndAfterAll {
  val log = LoggerFactory.getLogger(this.getClass)
  var sc : SparkContext = null
  var conf : Config = null

  override def beforeAll() = {
    sc = new SparkContext(new SparkConf().setAppName("caffe-on-spark").setMaster("local[4]"))

    val ROOT_PATH = {
      val fullPath = getClass.getClassLoader.getResource("log4j.properties").getPath
      fullPath.substring(0, fullPath.indexOf("caffe-grid/"))
    }
    val solver_config_path = ROOT_PATH + "caffe-grid/src/test/resources/lenet_memory_solver.prototxt";
    val args = Array("-conf", solver_config_path,
      "-model", "file:"+ROOT_PATH+"caffe-grid/target/mnistmodel")
    conf = new Config(sc, args)
  }

  override def afterAll() = {
    sc.stop()
  }

  test("Interleaving") {
    val caffeSpark = new CaffeOnSpark(sc)
    if (conf.solverParameter.hasTestInterval && conf.solverParameter.getTestIter(0) != 0) {
      caffeSpark.getResult = true
      log.info("interleave train and validation...")
      val sourceTrain: DataSource[Any,Any] = DataSource.getSource(conf, true).asInstanceOf[DataSource[Any, Any]]
      val sourceValidation: DataSource[Any,Any] = DataSource.getSource(conf, false).asInstanceOf[DataSource[Any, Any]]
      log.info("SolverParameter:"  + conf.solverParameter.getTestIter(0) + ":" + conf.solverParameter.hasTestInterval())
      caffeSpark.train(Array(sourceTrain, sourceValidation))
      var finalAccuracy: Float = 0
      var finalLoss: Float = 0
      for(i <- 0 until conf.solverParameter.getTestIter(0)){
        finalAccuracy += caffeSpark.interleaveResult(i)(0)
        finalLoss += caffeSpark.interleaveResult(i)(1)
      }
      assertTrue(finalAccuracy/conf.solverParameter.getTestIter(0) > 0.8)
      assertTrue(finalLoss/conf.solverParameter.getTestIter(0) < 0.5)
      assertEquals(caffeSpark.interleaveResult.length, conf.solverParameter.getTestIter(0))
      
    }
  }
}
