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
from pyspark.sql import SQLContext
import unittest

conf = SparkConf().setAppName("caffe-on-spark").setMaster("local")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

class PythonApiTest(unittest.TestCase):
    def grouper(self,iterable, n, fillvalue=None):
        args = [iter(iterable)] * n
        return izip_longest(fillvalue=fillvalue, *args)

    def setUp(self):
        #Initialize all objects
        self.cos=CaffeOnSpark(sc,sqlContext)
        cmdargs = conf.get('spark.pythonargs')
        args= dict(self.grouper(cmdargs.split(),2))
        self.cfg=Config(sc,args)
        self.dl_train_source = DataSource(sc).getSource(self.cfg,True)
        self.lr_raw_source = DataSource(sc).getSource(self.cfg,False)
        
    def testTrain(self):
        #Train
        self.cos.train(self.dl_train_source)

    def testTrainWithValidation(self):
        #TrainWithValidation
        result=self.cos.trainWithValidation(self.dl_train_source, self.lr_raw_source)
        self.assertEqual(self.cfg.solverParameter.getTestIter(0),len(result))
        finalAccuracy = 0
        finalLoss = 0
        for i in range(self.cfg.solverParameter.getTestIter(0)):
            finalAccuracy += result[i][0]
            finalLoss += result[i][1]

        self.assertTrue(finalAccuracy/self.cfg.solverParameter.getTestIter(0) > 0.8)
        self.assertTrue(finalLoss/self.cfg.solverParameter.getTestIter(0) < 0.5)      


    def testFeatures(self):
        #Extract features
        self.cos.features(self.lr_raw_source)


unittest.main(verbosity=2)            


