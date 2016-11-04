// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
package com.yahoo.ml.caffe.tools
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import scala.collection.Map
import java.io._

private [tools] object Vocab {
  val UNKNOWN_TOKEN = "UNK"
  val UNKNOWN_ID = 1
  val START_END_ID = 0
  val VALID_TOKEN_INDEX = 1
  val NO_TOKEN = -1
}
private[tools] class Vocab(val sqlContext: SQLContext) {

  private var map_word_index = scala.collection.mutable.Map[String, Int]()
  private var map_index_word = scala.collection.mutable.Map[Int, String]()
  private var rdd_word:RDD[Row] = null

  /* From a given dataset (RDD of sentences), we will construct a vocab.
  If vocabSize>0, we will select the top words per counts.
  */
  def genFromData(dataset: DataFrame, columnName: String, vocabSize: Int = -1): Unit = {
    synchronized {
      val rdd_words = dataset.select(columnName).rdd.flatMap{case Row(sentence:String) => Conversions.sentence2Words(sentence)}
      val rdd_sorted_vocab = rdd_words.map(word => (word, 1)).reduceByKey(_ + _).sortBy(-_._2)
      var cutoffVocabSize = rdd_sorted_vocab.count()
      if (vocabSize > 0)
        cutoffVocabSize = vocabSize
      // Embedding the word UNK
      var sorted_vocab_zipped = ((Vocab.UNKNOWN_TOKEN, Vocab.NO_TOKEN) +: (rdd_sorted_vocab.take(cutoffVocabSize.toInt))).zipWithIndex
      var rdd_sorted_vocab_zipped = sqlContext.sparkContext.parallelize(sorted_vocab_zipped)
      //Increment index by 2.
      // Number 0 is not used (as it indicates start or end of sentence)
      rdd_word = rdd_sorted_vocab_zipped.map(word_count_index =>
        Row(word_count_index._1._1))
    }
  }

  def save(vocabFilePath: String): Unit = {
    synchronized {
      rdd_word.map(word => word.getString(0)).coalesce(1, true).saveAsTextFile(vocabFilePath)
    }
  }

  def load(vocabFilePath: String): Unit = {
    synchronized {
      rdd_word = sqlContext.sparkContext.textFile(vocabFilePath).map(word => Row(word))
    }
    val map_word_index_immutable:Map[String,Int] = rdd_word.zipWithIndex().
      map(word_index => (word_index._1.getString(0), word_index._2.toInt)).collectAsMap()
    map_word_index ++= map_word_index_immutable
    map_index_word = map_word_index.map{case (k,v) => (v, k)}
  }

  def word2indexMap(): scala.collection.Map[String, Int] = map_word_index
  def index2wordMap(): scala.collection.Map[Int, String] = map_index_word
}

