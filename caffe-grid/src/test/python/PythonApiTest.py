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
import unittest
import os.path

conf = SparkConf().setAppName("caffe-on-spark").setMaster("local[1]")
sc = SparkContext(conf=conf)

class PythonApiTest(unittest.TestCase):
    def grouper(self,iterable, n, fillvalue=None):
        args = [iter(iterable)] * n
        return izip_longest(fillvalue=fillvalue, *args)

    def setUp(self):
        #Initialize all objects
        self.cos=CaffeOnSpark(sc)
        cmdargs = conf.get('spark.pythonargs')
        self.args= dict(self.grouper(cmdargs.split(),2))
        self.cfg=Config(sc,self.args)
        self.train_source = DataSource(sc).getSource(self.cfg,True)
        self.validation_source = DataSource(sc).getSource(self.cfg,False)
        
    def testTrain(self):
        self.cos.train(self.train_source)
        self.assertTrue(os.path.isfile(self.args.get('-model').split(":")[1][3:]))
        result=self.cos.features(self.validation_source)
        self.assertTrue('accuracy' in result.columns)
        self.assertTrue('ip1' in result.columns)
        self.assertTrue('ip2' in result.columns)
        self.assertTrue(result.count() > 100)
        self.assertTrue(result.first()['SampleID'] == '00000000')
        result=self.cos.test(self.validation_source)
        self.assertTrue(result.get('accuracy') > 0.9)

    def testTrainWithValidation(self):
        result=self.cos.trainWithValidation(self.train_source, self.validation_source)
        self.assertEqual(len(result.columns), 2)
        self.assertEqual(result.columns[0], 'accuracy')
        self.assertEqual(result.columns[1], 'loss')
        result.show(2)

        row_count = result.count()
        last_row = result.rdd.zipWithIndex().filter(lambda (row,index): index==(row_count - 1)).collect()[0][0]
        finalAccuracy = last_row[0][0]
        self.assertTrue(finalAccuracy > 0.8)
        finalLoss = last_row[1][0]
        self.assertTrue(finalLoss < 0.5)


unittest.main(verbosity=2)            


