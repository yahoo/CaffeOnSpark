'''
Copyright 2016 Yahoo Inc.
Licensed under the terms of the Apache 2.0 license.
Please see LICENSE file in the project root for terms.
'''

from ConversionUtil import wrapClass
from RegisterContext import registerContext

class DataSource:
    """Base class for various data sources.
       Each subclass must have a constructor with the following signature: (conf: Config, layerId: Int, isTrain: Boolean). 
       This is required by CaffeOnSpark at startup.

       :ivar SparkContext sc: The spark context of the current spark session
    """
    def __init__(self,sc):
        registerContext(sc)
        self.dataSource=wrapClass("com.yahoo.ml.caffe.DataSource")

    def getSource(self, conf, isTraining):
        """Returns a DataSource which can be used to train, test or extract features

        :param Config conf: Config object with details of datasource file location, devices, model file path and other relevant configurations
        :param Boolean isTraining: True for training and False for Test or feature extraction
        :rtype: DataSource
        :returns: a DataSource object
        """
        return self.dataSource.getSource(conf.__dict__['config'],isTraining)

    
