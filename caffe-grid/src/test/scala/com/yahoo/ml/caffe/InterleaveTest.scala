// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row}
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.slf4j.LoggerFactory
import org.testng.Assert._
import caffe.Caffe._
import com.yahoo.ml.jcaffe._

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
      log.info("interleave train and validation...")
      val sourceTrain: DataSource[Any,Any] = DataSource.getSource(conf, true).asInstanceOf[DataSource[Any, Any]]
      val sourceValidation: DataSource[Any,Any] = DataSource.getSource(conf, false).asInstanceOf[DataSource[Any, Any]]
      log.info("SolverParameter:"  + conf.solverParameter.getTestIter(0) + ":" + conf.solverParameter.hasTestInterval())
      val interleaveResult : DataFrame = caffeSpark.trainWithValidation(sourceTrain, sourceValidation)
      assertEquals(interleaveResult.columns, 2)
      val test_iter = conf.solverParameter.getTestIter(0)
      assertEquals(interleaveResult.count() % test_iter, 0)

      val (finalAccuracy: Float, finalLoss: Float) = interleaveResult.rdd.zipWithIndex().map{
        case (row:Row, index:Long) => {
          if (index >= (interleaveResult.count() - test_iter))
            (row.getFloat(0), row.getFloat(1))
          else
            (0.0F, 0.0F)
        }
      }.reduce{(x,y) => (x._1 + y._1, x._2 + y._2)}
      assertTrue(finalAccuracy/test_iter > 0.8)
      assertTrue(finalLoss/test_iter < 0.5)
    }
  }
}
