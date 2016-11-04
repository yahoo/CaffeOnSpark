// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.

package com.yahoo.ml.caffe.tools


import com.yahoo.ml.caffe.Config
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.slf4j.{Logger, LoggerFactory}

object CocoDataSetConverter {
  val log: Logger = LoggerFactory.getLogger(this.getClass)

  def main(args: Array[String]): Unit = {
    val sc_conf = new SparkConf()
    val sc = new SparkContext(sc_conf)
    val conf = new Config(sc, args)
    if (conf.imageRoot.length == 0 || conf.captionFile.length == 0) {
      log.error("Both -imageRoot and -captionFile must be defined")
    }
    val sqlContext = new SQLContext(sc)
    val df_image_caption = Conversions.Coco2ImageCaptionFile(sqlContext, conf.captionFile, conf.clusterSize)
    //wrte result DF
    if (!conf.imageCaptionDFDir.isEmpty)
      df_image_caption.write.json(conf.outputPath + conf.imageCaptionDFDir)

    val df_embedding =
      if (df_image_caption.columns.contains("caption")) {
        val vocab: Vocab = new Vocab(sqlContext)

        val fs: FileSystem = FileSystem.get(sc.hadoopConfiguration)
        if (!fs.exists(new Path(conf.outputPath + conf.vocabDir))) {
          vocab.genFromData(df_image_caption, "caption", conf.vocabSize)
          vocab.save(conf.outputPath + conf.vocabDir)
        }
        vocab.load(conf.outputPath + conf.vocabDir)

        val map_word_index: scala.collection.Map[String, Int] = vocab.word2indexMap
        Conversions.ImageCaption2Embedding(conf.imageRoot, df_image_caption, vocab, conf.captionLength)
      } else {
        Conversions.Image2Embedding(conf.imageRoot, df_image_caption)
      }

    df_embedding.write.format(conf.outputFormat).save(conf.outputPath + conf.embeddingDFDir)
  }
}
