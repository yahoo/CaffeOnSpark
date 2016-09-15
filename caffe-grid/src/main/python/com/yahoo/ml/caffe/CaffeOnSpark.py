'''
Copyright 2016 Yahoo Inc.
Licensed under the terms of the Apache 2.0 license.
Please see LICENSE file in the project root for terms.
'''

from ConversionUtil import wrapClass
from RegisterContext import registerContext
from pyspark.sql import DataFrame,SQLContext

class CaffeOnSpark:
    """CaffeOnSpark is the main class for distributed deep learning. 
    It will launch multiple Caffe cores within Spark executors, and conduct coordinated learning from HDFS datasets.

    :ivar SparkContext, SQLContext: The spark and sql context of the current spark session
    """

    def __init__(self,sc):
        registerContext(sc)
        spark_major_version = int(sc.version.split('.')[0])
        if spark_major_version >= 2:
            wrapClass("org.apache.spark.sql.Dataset")
        else:
            wrapClass("org.apache.spark.sql.DataFrame")
        self.__dict__['caffeonspark']=wrapClass("com.yahoo.ml.caffe.CaffeOnSpark")
        self.__dict__['cos']=self.__dict__.get('caffeonspark')(sc)
        self.__dict__['sqlcontext']=SQLContext(sc,self.__dict__['cos'].sqlContext)

    def train(self,train_source):
        """Training with a specific data source

        :param DataSource: the source for training data
        """
        self.__dict__.get('cos').train(train_source)

    def test(self,test_source):
        """Test with a specific data source.

        :param DataSource: the source for the test data
        """
        return self.__dict__.get('cos').test(test_source)

    def features(self,source):
        """Extract features from a specific data source.

        :param DataSource: the features to extract
        """
        extracted_df = self.__dict__.get('cos').features(source)
        extracted_pydf = DataFrame(extracted_df.javaInstance,self.__dict__.get('sqlcontext'))
        return extracted_pydf

    def trainWithValidation(self,train_source, validation_source):
        """Training with interleaved validation

        :param DataSource: source for training data
        :param DataSource: source for validation data
        """
        validation_df = self.__dict__.get('cos').trainWithValidation(train_source, validation_source)
        validation_pydf = DataFrame(validation_df.javaInstance,self.__dict__.get('sqlcontext'))
        return validation_pydf
    
