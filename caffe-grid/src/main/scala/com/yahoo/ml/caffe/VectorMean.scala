// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe

import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types._

/**
 * VectorMean computes the element-wise mean of vectors in a dataframe.
 *
 * @param n the length of the vector
 *
 */

private[caffe] class VectorMean (n: Int) extends UserDefinedAggregateFunction {

  // Input Data Type Schema
  // input is the a vector of floats
  def inputSchema: StructType = StructType(Array(StructField("name", ArrayType(FloatType), false)))

  // Intermediate Schema
  // the partial sum is saved as a vector of doubles.
  def bufferSchema = StructType(Array(
    StructField("sum", ArrayType(DoubleType), false),
    StructField("cnt", LongType, false)
  ))

  // Returned Data Type.
  // return vector of doubles.
  def dataType: DataType = ArrayType(DoubleType)

  // Self-explaining
  def deterministic = true

  // This function is called whenever key changes
  def initialize(buffer: MutableAggregationBuffer) = {
    buffer(0) = Array.fill(n)(0.0)
    buffer(1) = 0L
  }

  // Iterate over each entry of a group
  def update(buffer: MutableAggregationBuffer, input: Row) = {
    buffer(0) = (0 until n).map(i => buffer.getSeq[Double](0)(i) + input.getSeq[Float](0)(i)).toArray
    buffer(1) = buffer.getLong(1) + 1
  }

  // Merge two partial aggregates
  def merge(buffer1: MutableAggregationBuffer, buffer2: Row) = {
    buffer1(0) = (0 until n).map(i => buffer1.getSeq[Double](0)(i) + buffer2.getSeq[Double](0)(i)).toArray
    buffer1(1) = buffer1.getLong(1) + buffer2.getLong(1)
  }

  // Called after all the entries are exhausted.
  def evaluate(buffer: Row) = {
    val vmean = Array.fill(n)(0.0)
    for (i <- 0 until n) {
      vmean(i) = buffer.getSeq[Double](0)(i)/buffer.getLong(1).toDouble
    }
    vmean
  }

}
