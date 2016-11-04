// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe.tools

import java.io.ByteArrayOutputStream

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, FSDataInputStream, Path}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.WrappedArray
import scala.util.matching.Regex


object Conversions {
  val log: Logger = LoggerFactory.getLogger(this.getClass)

  def sentence2Words(sentence: String): Array[String] = {
    val pattern = new Regex("(\\w+|\\W+)")
    val res = (pattern findAllIn sentence.trim).filter(c => c.trim.length > 0).toArray.map(x => x.toLowerCase().trim)
    if (res.last != ".")
      res
    else
      res.slice(0,res.length-1)
  }

  def Coco2ImageCaptionFile(sqlContext: SQLContext, src: String, clusterSize: Int): DataFrame = {
    //read input DF
    val df = sqlContext.read.json(src)

    //annotation RDD
    val rdd_captions_ids: RDD[(Long, (Long, String))] =
      if (!df.columns.contains("annotations")) null
      else
        df.select("annotations.image_id", "annotations.id", "annotations.caption").rdd.
          coalesce(clusterSize, true).flatMap {
          case Row(image_ids: WrappedArray[Any], ids: WrappedArray[Any], captions: WrappedArray[Any]) => {
            val len = ids.length
            val caption_arr = new Array[(Long, (Long, String))](len)
            for (i <- 0 until ids.length)
              caption_arr(i) = (image_ids(i).asInstanceOf[Long],
                (ids(i).asInstanceOf[Long], captions(i).asInstanceOf[String]))
            caption_arr.toIterator
          }
        }


    //image RDD
    val url_column = if (rdd_captions_ids != null) "images.flickr_url" else "images.coco_url"
    val rdd_images: RDD[(Long, (Int, Int, String, String))] = df.select("images.id", "images.height", "images.width",
      "images.file_name", url_column).rdd.coalesce(clusterSize, true).flatMap {
      case Row(ids: WrappedArray[Any], heights: WrappedArray[Any], widths: WrappedArray[Any],
      files: WrappedArray[Any], urls: WrappedArray[Any]) => {
        val len = ids.length
        val image_arr = new Array[(Long, (Int, Int, String, String))](len)
        for (i <- 0 until ids.length)
          image_arr(i) = (ids(i).asInstanceOf[Long],
            (heights(i).asInstanceOf[Long].toInt, widths(i).asInstanceOf[Long].toInt,
              files(i).asInstanceOf[String], urls(i).asInstanceOf[String]))
        image_arr.toIterator
      }
    }

    //result schema
    var schema = new StructType(Array(StructField("id", LongType, true),
      StructField("height", IntegerType, true),
      StructField("width", IntegerType, true),
      StructField("file", StringType, true),
      StructField("url", StringType, true)))
    if (rdd_captions_ids != null)
      schema = schema.add("caption", StringType, true)

    //result data frame
    val result_rdd =
      if (rdd_captions_ids != null)
        rdd_images.join(rdd_captions_ids).map(x => Row(x._2._2._1, //caption ID
          x._2._1._1, x._2._1._2, x._2._1._3, x._2._1._4, x._2._2._2))
      else
        rdd_images.map(x => Row(x._1, x._2._1, x._2._2, x._2._3, x._2._4))
    val result_df = sqlContext.createDataFrame(result_rdd, schema)

    result_df
  }

  val INNER_DATA_FIELD_SCHEMA = StructType(Array(
    StructField("height", IntegerType, true),
    StructField("width", IntegerType, true),
    StructField("image", ArrayType(ByteType), true)))

  private def image2innerRow(fs: FileSystem, file: String, height: Int, width: Int, buffer: Array[Byte]): Row = {
    val bout = new ByteArrayOutputStream
    val in: FSDataInputStream = fs.open(new Path(file))
    var len = in.read(buffer, 0, buffer.length)
    while (len > 0) {
      bout.write(buffer, 0, len)
      len = in.read(buffer, 0, buffer.length)
    }
    in.close()

    Row(height, width, bout.toByteArray)
  }

  def Image2Embedding(imageRootFolder: String, imageCaptionDF: DataFrame): DataFrame = {
    val rdd_id_imagefile_ht_wt = imageCaptionDF.select("id", "file", "height", "width").rdd.map {
      case Row(id: Long, imageFileName: String, height: Long, width: Long) =>
        (id, imageFileName, height.toInt, width.toInt)
      case Row(id: Long, imageFileName: String, height: Int, width: Int) =>
        (id, imageFileName, height, width)
    }
    //for each row of image and caption, produce a row of embedded rdd
    val rdd_embedding = rdd_id_imagefile_ht_wt.mapPartitions { iter => {
      //Setup for image embedding
      val fs = new Path(imageRootFolder).getFileSystem(new Configuration)
      val log: Logger = LoggerFactory.getLogger(this.getClass)
      val buffer = new Array[Byte](1024 * 1024)

      iter.map { case (id: Long, imageFileName: String, height: Int, width: Int) =>
        try {
          Row(id.toString,
            image2innerRow(fs, imageRootFolder + "/" + imageFileName, height, width, buffer),
            0)
        } catch {
          case e: Exception => {
            log.warn(imageFileName + " read exception, and will be skipped")
            throw e
          }
        }
      }
    }
    }

    val schema = StructType(Array(
      StructField("id", StringType, true),
      StructField("data", INNER_DATA_FIELD_SCHEMA, true),
      StructField("label", IntegerType, true)))

    val dataframe_embedding = imageCaptionDF.sqlContext.createDataFrame(rdd_embedding, schema)
    dataframe_embedding
  }


  def ImageCaption2Embedding(imageRootFolder: String,
                             imageCaptionDF: DataFrame, vocab: Vocab, captionLength: Int): DataFrame = {
    val map_word_index: scala.collection.Map[String, Int] = vocab.word2indexMap
    val rdd_id_imagefile_ht_wt_caption = imageCaptionDF.select("id", "file", "height", "width", "caption").rdd.map {
      case Row(id: Long, imageFileName: String, height: Long, width: Long, caption: String) =>
        (id, imageFileName, height.toInt, width.toInt, caption)
      case Row(id: Long, imageFileName: String, height: Int, width: Int, caption: String) =>
        (id, imageFileName, height, width, caption)
    }
    //for each row of image and caption, produce a row of embedded rdd
    val rdd_embedding = rdd_id_imagefile_ht_wt_caption.mapPartitions { iter => {
      //Setup for image embedding
      val fs = new Path(imageRootFolder).getFileSystem(new Configuration)
      val log: Logger = LoggerFactory.getLogger(this.getClass)
      val buffer = new Array[Byte](1024 * 1024)

      iter.map { case (id: Long, imageFileName: String, height: Int, width: Int, caption: String) =>
        try {
          //generate caption embedding
          val inputSentenceEmbedding: Array[Int] = Array.fill[Int](captionLength)(-1)
          val targetSentenceEmbedding: Array[Int] = Array.fill[Int](captionLength)(-1)
          val contSentenceEmbedding: Array[Int] = Array.fill[Int](captionLength)(-1)
          val words = Conversions.sentence2Words(caption)
          val len = if (words.length < captionLength) words.length else captionLength-1
          inputSentenceEmbedding(0) = Vocab.START_END_ID
          contSentenceEmbedding(0) = 0
          for (i <- 0 until len) {
            var embedding = Vocab.UNKNOWN_ID
            if (map_word_index.contains(words(i)))
              embedding = map_word_index(words(i)).toInt

            inputSentenceEmbedding(i + 1) = embedding
            contSentenceEmbedding(i + 1) = 1
            targetSentenceEmbedding(i) = embedding
          }
          targetSentenceEmbedding(len) = Vocab.START_END_ID

          Row(id.toString,
            image2innerRow(fs, imageRootFolder + "/" + imageFileName, height, width, buffer),
            0,
            inputSentenceEmbedding, contSentenceEmbedding, targetSentenceEmbedding)
        } catch {
          case e: Exception => {
            log.warn(imageFileName + " read exception, and will be skipped")
            throw e
          }
        }
      }
    }
    }

    val schema = StructType(Array(
      StructField("id", StringType, true),
      StructField("data", INNER_DATA_FIELD_SCHEMA, true),
      StructField("label", IntegerType, true),
      StructField("input_sentence", ArrayType(IntegerType), true),
      StructField("cont_sentence", ArrayType(IntegerType), true),
      StructField("target_sentence", ArrayType(IntegerType), true)))

    val dataframe_embedding = imageCaptionDF.sqlContext.createDataFrame(rdd_embedding, schema)
    dataframe_embedding
  }

  def Embedding2Caption(embeddingDF: DataFrame, vocab: Vocab,
                        embeddedingColumn: String, captionColumn: String): DataFrame = {
    val map_index_word: scala.collection.Map[Int, String] = vocab.index2wordMap
    val rdd_id_captions = embeddingDF.select("id", embeddedingColumn).rdd.map { case Row(id: String, embedding_array: WrappedArray[Any]) => {
      var caption = new StringBuilder
      embedding_array.map(embedding_index =>
        if (embedding_index != Vocab.START_END_ID && embedding_index != Vocab.NO_TOKEN) {
          caption.append(map_index_word(embedding_index.asInstanceOf[Int]).toString)
          caption.append(" ")
        })
      caption.deleteCharAt(caption.length - 1)
      Row(id, caption.toString())
    }
    }
    val schema = StructType(Array(
      StructField("id", LongType, true),
      StructField(captionColumn, StringType, true)))

    val dataframe_id_caption = embeddingDF.sqlContext.createDataFrame(rdd_id_captions, schema)
    dataframe_id_caption
  }
}
