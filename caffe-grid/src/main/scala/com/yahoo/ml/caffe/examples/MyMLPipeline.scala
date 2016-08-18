// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe.examples

import com.yahoo.ml.caffe.{Config, CaffeOnSpark, DataSource}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.classification.LogisticRegression
import scala.collection.immutable.Map
/**
 * Sample Spark program that uses
 * CaffeOnSpark for deep learning, and
 * MLlib for conventional machine learning
 */
object MyMLPipeline {
  def main(args: Array[String]): Unit = {
    //CaffeOnSpark initialization
    val ctx = new SparkContext(new SparkConf())
    val cos = new CaffeOnSpark(ctx)
    var conf = new Config(ctx, args)

    //perform DL training using the TRAINING source specified in Net prototxt
    val dl_train_source = DataSource.getSource(conf, true)
    cos.train(dl_train_source)

    //apply DL model for feature extraction using the TEST source specified in Net prototxt
    val lr_raw_source = DataSource.getSource(conf, false)
    val extracted_df = cos.features(lr_raw_source)

    //prepare data for MLLib LogisticRegression
    val lr_input_df = extracted_df.withColumn("Label", cos.floatarray2doubleUDF(extracted_df(conf.label)))
                      .withColumn("Feature", cos.floatarray2doublevectorUDF(extracted_df(conf.features(0))))

    //Learn a LogisticRegression model via Apache MLlib
    val lr = new LogisticRegression().setLabelCol("Label").setFeaturesCol("Feature")
    val lr_model = lr.fit(lr_input_df)

    //save the LogisticRegression classification model onto HDFS
    lr_model.write.overwrite().save(conf.outputPath)
  }
}
