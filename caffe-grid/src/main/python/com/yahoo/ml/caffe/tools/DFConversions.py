'''
Copyright 2016 Yahoo Inc.
Licensed under the terms of the Apache 2.0 license.
Please see LICENSE file in the project root for terms.
'''
from PIL import Image
from io import BytesIO
from IPython.display import HTML
import numpy as np
from base64 import b64encode
from google.protobuf import text_format
import array
from com.yahoo.ml.caffe.ConversionUtil import wrapClass, getScalaSingleton, toPython
from com.yahoo.ml.caffe.RegisterContext import registerContext
from pyspark.sql import DataFrame,SQLContext

class DFConversions:
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
        df = self.__dict__.get('conversions').Coco2ImageCaptionFile(self.__dict__.get('sqlContext'), src, clusterSize)
        pydf = DataFrame(df,self.__dict__.get('sqlContext'))
        return pydf


    def Image2Embedding(self, imageRootFolder, imageCaptionDF):
        """Get the embedding for the image as a dataframe
        :param imageRootFolder: the src folder of the images
        :param imageCaptionDF: the dataframe with the image file and image attributes
        """
        df = self.__dict__.get('conversions').Image2Embedding(imageRootFolder, imageCaptionDF._jdf)
        pydf = DataFrame(df,self.__dict__.get('sqlContext'))
        return pydf

    def ImageCaption2Embedding(self, imageRootFolder, imageCaptionDF, vocab, captionLength):
        """Get the embedding for the images as well as the caption as a dataframe
        :param imageRootFolder: the src folder of the images
        :param imageCaptionDF: the dataframe with the images as well as captions
        :param vocab: the vocab object
        :param captionLength: Length of the embedding to generate for the caption
        """
        df = self.__dict__.get('conversions').ImageCaption2Embedding(imageRootFolder, imageCaptionDF._jdf, vocab.vocabObject, captionLength)
        pydf = DataFrame(df,self.__dict__.get('sqlContext'))
        return pydf


    def Embedding2Caption(self, embeddingDF, vocab, embeddingColumn, captionColumn):
        """Get the captions from the embeddings
        :param embeddingDF: the dataframe which contains the embedding
        :param vocab: the vocab object
        :param embeddingColumn: the embedding column name in embeddingDF which contains the caption embedding
        """
        df = self.__dict__.get('conversions').Embedding2Caption(embeddingDF._jdf, vocab.vocabObject, embeddingColumn, captionColumn)
        pydf = DataFrame(df,self.__dict__.get('sqlContext'))
        return pydf


def get_image(image):
    bytes = array.array('b', image)
    return "<img src='data:image/png;base64," + b64encode(bytes) + "' />"


def show_captions(df, nrows=10):
    """Displays a table of captions(both original as well as predictions) with their images, inline in html

        :param DataFrame df: A python dataframe
        :param int nrows: First n rows to display from the dataframe
    """
    data = df.take(nrows)
    html = "<table><tr><th>Image Id</th><th>Image</th><th>Prediction</th>"
    for i in range(nrows):
        row = data[i]
        html += "<tr>"
        html += "<td>%s</td>" % row.id
        html += "<td>%s</td>" % get_image(row.data.image)
        html += "<td>%s</td>" % row.prediction
        html += "</tr>"
    html += "</table>"
    return HTML(html)
