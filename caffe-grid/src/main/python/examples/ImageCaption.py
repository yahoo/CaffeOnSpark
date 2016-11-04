# Copyright 2016 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.
import caffe
from examples.coco.retrieval_experiment import *
from pyspark.sql import SQLContext
from pyspark import SparkConf,SparkContext
from pyspark.sql.types import *
from itertools import izip_longest
import json
import argparse

def predict_caption(list_of_images, model, imagenet, lstmnet, vocab):
  out_iterator = []
  ce = CaptionExperiment(str(model),str(imagenet),str(lstmnet),str(vocab))
  for image in list_of_images:
    out_iterator.append(ce.getCaption(image))
  return iter(out_iterator)

def get_predictions(sqlContext, images, model, imagenet, lstmnet, vocab):
  rdd = images.mapPartitions(lambda im: predict_caption(im, model, imagenet, lstmnet, vocab))
  INNERSCHEMA = StructType([StructField("id", StringType(), True),StructField("prediction", StringType(), True)])
  schema = StructType([StructField("result", INNERSCHEMA, True)])
  return sqlContext.createDataFrame(rdd, schema).select("result.id", "result.prediction")

def main():
  conf = SparkConf()
  sc = SparkContext(conf=conf)
  sqlContext = SQLContext(sc)
  cmdargs = conf.get('spark.pythonargs')
  parser = argparse.ArgumentParser(description="Image to Caption Util")
  parser.add_argument('-input', action="store", dest="input")
  parser.add_argument('-model', action="store", dest="model")
  parser.add_argument('-imagenet', action="store", dest="imagenet")
  parser.add_argument('-lstmnet', action="store", dest="lstmnet")
  parser.add_argument('-vocab', action="store", dest="vocab")
  parser.add_argument('-output', action="store", dest="output")
  
  args=parser.parse_args(cmdargs.split(" "))

  df_input = sqlContext.read.parquet(str(args.input))
  images = df_input.select("data.image","data.height", "data.width", "id")
  df=get_predictions(sqlContext, images, str(args.model), str(args.imagenet), str(args.lstmnet), str(args.vocab))
  df.write.json(str(args.output))


if __name__ == "__main__":
    main()


