// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import org.apache.spark.{SparkConf, SparkContext}

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.types._
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.slf4j.LoggerFactory
import org.testng.Assert._


class DataFrameTest extends FunSuite with BeforeAndAfterAll {
  val log = LoggerFactory.getLogger(this.getClass)
  var sc : SparkContext = null
  var sqlContext: SQLContext = null

  override def beforeAll() = {
    sc = new SparkContext(new SparkConf().setAppName("caffe-on-spark").setMaster("local"))
    sqlContext = new org.apache.spark.sql.SQLContext(sc)
  }

  override def afterAll() = {
    sc.stop()
  }

  test("Vector Mean Test") {
    // generate schema
    val schema =
      StructType(Array(StructField("name", StringType, false), StructField("data", ArrayType(FloatType), false)))

    // dataset, 3 rows 2 columns, the 2nd column is an array of 2 elements
    val a:Array[Float] = Array(20.37F, 56.87F, 123.56F)
    val b:Array[Float] = Array(-11.34F, 45.89F, 127.30F)
    var people = Array(Array("John", Array(a(0), b(0))))
    people = people :+ Array("Tom", Array(a(1), b(1)))
    people = people :+ Array("Kate", Array(a(2), b(2)))

    // form the rdd
    val dataRDD = sc.parallelize(people)

    // Convert records of the RDD (people) to Rows.
    val rowRDD = dataRDD.map(p => Row.fromSeq(p))

    // Create the dataframe based on the schema.
    val df = sqlContext.createDataFrame(rowRDD, schema)

    // take the element-wise mean of the vectors
    val meanDF = df.agg(new VectorMean(2)(df("data")))

    // get the result
    val r: Seq[Double] = meanDF.take(1)(0).getSeq[Double](0)

    assertEquals(r(0), a.sum/a.size, 1.0e-5)
    assertEquals(r(1), b.sum/b.size, 1.0e-5)
  }

}
