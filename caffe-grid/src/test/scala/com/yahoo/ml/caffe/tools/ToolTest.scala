// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe.tools

import java.io.File
import java.net.URL

import com.yahoo.ml.caffe.Config
import org.apache.commons.io.FileUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfterAll, FunSuite}
import org.slf4j.{Logger, LoggerFactory}
import org.testng.Assert._

import scala.collection.mutable
import sys.process._

class ToolTest extends FunSuite with BeforeAndAfterAll {
  val log: Logger = LoggerFactory.getLogger(this.getClass)
  var sc : SparkContext = null

  override def beforeAll() = {
    sc = new SparkContext(new SparkConf().setAppName("caffe-on-spark").setMaster("local"))
  }

  override def afterAll() = {
    sc.stop()
  }

  test("Binary2Sequence") {
    val ROOT_PATH = {
      val fullPath = getClass.getClassLoader.getResource("log4j.properties").getPath
      fullPath.substring(0, fullPath.indexOf("caffe-grid/"))
    }
    val args = Array(
      "-imageRoot", "file:"+ROOT_PATH+"data/images",
      "-labelFile", "file:"+ROOT_PATH+"data/images/labels.txt",
      "-output", "file:"+ROOT_PATH+"caffe-grid/target/seq_files"
    )
    val conf = new Config(sc, args)

    val path = new Path(conf.outputPath)
    val fs = path.getFileSystem(new Configuration)
    if (fs.exists(path)) fs.delete(path, true)

    val rdd = new Binary2Sequence(sc, conf).makeRDD()

    //check file size
    assertEquals(rdd.count(), sc.textFile(conf.labelFile).count())
  }

  private def downloadImageDataSet(dataSetFolder: String, fileNameURL: RDD[Row]):Unit = {
    fileNameURL.map{ case Row(file:String, url:String) =>
      new URL(url) #> new File(dataSetFolder + "/" + file) !!
    }.collect()
  }

  private def inputDF2PairRDD(dataframe: DataFrame) : RDD[(Long, (Int, Int, String))] =
    dataframe.select("id", "height", "width", "file").rdd.map{
      case Row(id:Long, height:Int, width:Int, file:String)
      => (id.toLong, (height, width, file))}

  private def embeddingDF2PairRDD(dataframe: DataFrame) : RDD[(Long, (Int, Int, mutable.WrappedArray[Byte]))] =
    dataframe.select("id", "data.height","data.width","data.image").rdd.map{
      case Row(id:String, height:Int, width:Int, image:mutable.WrappedArray[Byte])
      => (id.toLong, (height, width, image))}

  private def assertImageEmbeddings(df_embedding: DataFrame, inputRdd: RDD[(Long, (Int, Int, String))],
                                    cocoImageRoot: String): Unit =
    embeddingDF2PairRDD(df_embedding).join(inputRdd).map { case (id: Long,
    ((height1: Int, width1: Int, image: mutable.WrappedArray[Byte]),
    (height2: Int, width2: Int, file: String))) => {
      assertEquals(height1, height2)
      assertEquals(width1, width2)
      assertEquals(image.size, new File(cocoImageRoot + "/" + file).length())
    }
    }.collect()

  test("CocoTest") {
    val ROOT_PATH = {
      val fullPath = getClass.getClassLoader.getResource("log4j.properties").getPath
      fullPath.substring(0, fullPath.indexOf("caffe-grid/"))+"caffe-grid/"
    }
    val cocoJson = ROOT_PATH+"src/test/resources/coco.json"
    val cocoImageRoot = ROOT_PATH+"src/test/resources/"
    val cocoImageCaptionDF = ROOT_PATH+"target/coco_df_image_caption"
    val cocoVocab = ROOT_PATH+"target/coco_vocab"
    val cocoEmbeddingDF = ROOT_PATH+"target/coco_df_embedding"

    val sqlContext = new SQLContext(sc)

    FileUtils.deleteQuietly(new File(cocoImageCaptionDF))
    val df_image_caption = Conversions.Coco2ImageCaptionFile(sqlContext, cocoJson, 4)
    val count = df_image_caption.count.toInt

    FileUtils.deleteQuietly(new File(cocoEmbeddingDF))
    FileUtils.deleteQuietly(new File(cocoVocab))

    val vocab:Vocab = new Vocab(sqlContext)
    vocab.genFromData(df_image_caption, "caption", 23)
    vocab.save(cocoVocab)
    
    vocab.load(cocoVocab)
    val map_word_index:scala.collection.Map[String, Int] = vocab.word2indexMap
    assertTrue(map_word_index.size>10)
    assertTrue(map_word_index("butterfly")>Vocab.VALID_TOKEN_INDEX)
    
    val captionLength = 10
    val df_embedding = Conversions.ImageCaption2Embedding(cocoImageRoot, df_image_caption, vocab, captionLength)
    
    val df_source_captions = Conversions.Embedding2Caption(df_embedding, vocab, "input_sentence", "caption").select("caption")
    val df_target_captions = Conversions.Embedding2Caption(df_embedding, vocab, "target_sentence", "caption").select("caption")
    
    val input_captions :Array[String] = df_image_caption.select("caption").rdd.map{case Row(c:String) => c}.take(count)
    val source_captions :Array[String] = df_source_captions.rdd.map{case Row(c:String) => c}.take(count)
    val target_captions :Array[String] = df_target_captions.rdd.map{case Row(c:String) => c}.take(count)
    for (i <- 0 until count) {
      val input_caption_array = Conversions.sentence2Words(input_captions(i))
      var cutoff = input_caption_array.length
      if (cutoff >= captionLength)
        cutoff = captionLength-1
      //Test the source embedding
      assertEquals( input_caption_array.take(cutoff),
        Conversions.sentence2Words(source_captions(i)))
      //Test the target embedding
      assertEquals(input_caption_array.take(cutoff),
        Conversions.sentence2Words(target_captions(i)))
    }
  }
}
