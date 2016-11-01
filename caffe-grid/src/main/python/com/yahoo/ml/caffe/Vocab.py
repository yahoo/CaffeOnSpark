'''
Copyright 2016 Yahoo Inc.
Licensed under the terms of the Apache 2.0 license.
Please see LICENSE file in the project root for terms.
'''

from ConversionUtil import wrapClass
from RegisterContext import registerContext
from pyspark.sql import DataFrame,SQLContext

class Vocab:
    """

    :ivar SparkContext: The spark context of the current spark session
    """

    def __init__(self,sc):
        registerContext(sc)
        self.__dict__['vocab']=wrapClass("com.yahoo.ml.caffe.tools.Vocab")
        self.__dict__['sqlContext']=SQLContext(sc)
        self.__dict__['vocabObject']=self.__dict__['vocab'](self.__dict__['sqlContext'])

    def genFromData(self,dataset,columnName,vocabSize):
        """Convert generate the vocabulary from dataset
        :param dataset: dataframe containing the captions
        :param columnName: column in the dataset which has the caption
        :param vocabSize: Size of the vocabulary to generate (with vocab in descending order)
        """
        self.__dict__.get('vocabObject').genFromData(dataset,columnName,vocabSize)

    def save(self, vocabFilePath):
        """Save the generated vocabulary
        :param vocabFilePath: the name of the file to save the vocabulary to
        """
        self.__dict__.get('vocabObject').save(vocabFilePath)
        
    def load(self, vocabFilePath):
        """Load the vocabulary from a file
        :param vocabFilePath: the name of the file to load the vocabulary from
        """
        self.__dict__.get('vocabObject').load(vocabFilePath)


