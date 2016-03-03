package com.yahoo.ml.caffe.tools

import com.yahoo.ml.caffe.{LMDB, Config}
import org.apache.spark.{SparkContext, SparkConf}
import org.slf4j.{LoggerFactory, Logger}
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.sql.types.{StructField, StructType, ArrayType, IntegerType, StringType, BooleanType, BinaryType}
import scala.collection.JavaConversions

object LMDB2DataFrame {
  val log: Logger = LoggerFactory.getLogger(this.getClass)
  val schema = new StructType(Array(StructField("id", StringType, true),
    StructField("label", StringType, false),
    StructField("channels", IntegerType,  true),
    StructField("height", IntegerType,  true),
    StructField("width", IntegerType,  true),
    StructField("encoded", BooleanType,  true),
    StructField("data", BinaryType, false)))

  def main(args: Array[String]) {
    val sc_conf = new SparkConf()
    val sc = new SparkContext(sc_conf)

    //configure
    val conf = new Config(sc, args)
    if (conf.imageRoot.length == 0 || conf.outputPath.length == 0) {
      log.error("-imageRoot <LMDB directory> and -output <DataFrameFile> must be defined")
      return
    }

    //produce data frame
    val list = JavaConversions.seqAsJavaList(LMDB.makeRowSeq(LMDB.toLocalFile(conf.imageRoot)))
    log.info("List size:"+list.size())

    val sqlContext = new SQLContext(sc)
    val df : DataFrame = sqlContext.createDataFrame(list, schema)

    //save into output file on HDFS
    log.info("Saving DF at "+conf.outputPath)
    df.write.format(conf.outputFormat).save(conf.outputPath)
    sc.stop()
  }
}

