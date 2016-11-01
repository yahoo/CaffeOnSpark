'''
Copyright 2016 Yahoo Inc.
Licensed under the terms of the Apache 2.0 license.
Please see LICENSE file in the project root for terms.
'''

from ConversionUtil import wrapClass
from RegisterContext import registerContext
from pyspark.sql import DataFrame,SQLContext
from ConversionUtil import getScalaSingleton, toPython

class Conversions:
    """

    :ivar SparkContext: The spark context of the current spark session
    """

    def __init__(self,sc):
        registerContext(sc)
        wrapClass("com.yahoo.ml.caffe.tools.Conversions$")
        self.__dict__['conversions']=toPython(getScalaSingleton("com.yahoo.ml.caffe.tools.Conversions"))
        self.__dict__['sqlContext']=SQLContext(sc)

    def Coco2ImageCaptionFile(self,src,clusterSize):
        """Convert Cocodataset to Image Caption Dataframe
        :param src: the source for coco dataset i.e the caption file 
        :param clusterSize: No. of executors
        """
        self.__dict__.get('conversions').Coco2ImageCaptionFile(self.__dict__.get('sqlContext'), src, clusterSize)

    def Image2Embedding(self, imageRootFolder, imageCaptionDF):
        """Get the embedding for the image as a dataframe
        :param imageRootFolder: the src folder of the images
        :param imageCaptionDF: the dataframe with the image file and image attributes
        """
        self.__dict__.get('conversions').Image2Embedding(imageRootFolder, imageCaptionDF)

    def ImageCaption2Embedding(self, imageRootFolder, imageCaptionDF, vocab, captionLength):
        """Get the embedding for the images as well as the caption as a dataframe
        :param imageRootFolder: the src folder of the images
        :param imageCaptionDF: the dataframe with the images as well as captions
        :param vocab: the vocab object
        :param captionLength: Length of the embedding to generate for the caption
        """
        self.__dict__.get('conversions').ImageCaption2Embedding(imageRootFolder, imageCaptionDF, vocab, captionLength)

    def Embedding2Caption(self, embeddingDF, vocab, embeddingColumn, captionColumn):
        """Get the captions from the embeddings
        :param embeddingDF: the dataframe which contains the embedding
        :param vocab: the vocab object
        :param embeddingColumn: the embedding column name in embeddingDF which contains the caption embedding
        """
        self.__dict__.get('conversions').Embedding2Caption(embeddingDF, vocab, embeddingColumn, captionColumn)
