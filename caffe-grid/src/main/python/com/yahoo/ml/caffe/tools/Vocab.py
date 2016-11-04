'''
Copyright 2016 Yahoo Inc.
Licensed under the terms of the Apache 2.0 license.
Please see LICENSE file in the project root for terms.
'''

from com.yahoo.ml.caffe.ConversionUtil import wrapClass
from com.yahoo.ml.caffe.RegisterContext import registerContext
from pyspark.sql import DataFrame,SQLContext

class Vocab:
    """

    :ivar SparkContext: The spark context of the current spark session
    """

    def __init__(self,sc):
        registerContext(sc)
        self.vocab=wrapClass("com.yahoo.ml.caffe.tools.Vocab")
        self.sqlContext=SQLContext(sc)
        self.vocabObject=self.vocab(self.sqlContext)

    def genFromData(self,dataset,columnName,vocabSize):
        """Convert generate the vocabulary from dataset
        :param dataset: dataframe containing the captions
        :param columnName: column in the dataset which has the caption
        :param vocabSize: Size of the vocabulary to generate (with vocab in descending order)
        """
        self.vocabObject.genFromData(dataset._jdf,columnName,vocabSize)

    def save(self, vocabFilePath):
        """Save the generated vocabulary
        :param vocabFilePath: the name of the file to save the vocabulary to
        """
        self.vocabObject.save(vocabFilePath)
        
    def load(self, vocabFilePath):
        """Load the vocabulary from a file
        :param vocabFilePath: the name of the file to load the vocabulary from
        """
        self.vocabObject.load(vocabFilePath)


