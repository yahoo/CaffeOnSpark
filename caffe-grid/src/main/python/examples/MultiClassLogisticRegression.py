'''
Copyright 2016 Yahoo Inc.
Licensed under the terms of the Apache 2.0 license.
Please see LICENSE file in the project root for terms.
'''

from com.yahoo.ml.caffe.CaffeOnSpark import CaffeOnSpark
from com.yahoo.ml.caffe.Config import Config
from com.yahoo.ml.caffe.DataSource import DataSource
from pyspark.sql import DataFrame
from pyspark.mllib.linalg import Vectors
from pyspark.sql import Row
from pyspark import SparkConf,SparkContext
from itertools import izip_longest
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.sql import SQLContext

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

conf = SparkConf()
sc = SparkContext(conf=conf)
#Initialize all objects
cos=CaffeOnSpark(sc)
cmdargs = conf.get('spark.pythonargs')
args= dict(grouper(cmdargs.split(),2))
cfg=Config(sc,args)
dl_train_source = DataSource(sc).getSource(cfg,True)
#Train
cos.train(dl_train_source)
lr_raw_source = DataSource(sc).getSource(cfg,False)
#Extract features
extracted_df = cos.features(lr_raw_source)
# Do multiclass LogisticRegression
data = extracted_df.map(lambda row: LabeledPoint(row.label[0], Vectors.dense(row.ip1)))
lr = LogisticRegressionWithLBFGS.train(data, numClasses=10, iterations=10)
predictions = lr.predict(data.map(lambda pt : pt.features))
