'''
Copyright 2016 Yahoo Inc.
Licensed under the terms of the Apache 2.0 license.
Please see LICENSE file in the project root for terms.
'''

from ConversionUtil import wrapClass
from RegisterContext import registerContext

class Config:
    """CaffeOnSpark configuration

    :ivar SparkContext sc: The spark context of the current spark session
    :ivar dict args: Dictionary of configuration parameters (optional). 
    :ivar int clusterSize: Get/Set # of executors in the cluster
    :ivar int connection: Get/Set network interface for connection among Spark executors.
    :ivar int devices: Get/Set # of GPUs/CPUs per Spark executor
    :ivar Array[String] features: Get/Set blob names of feature output blobs
    :ivar String imageRoot: Get/Set the path of input image folder
    :ivar Boolean isFeature: True if we perform feature extraction
    :ivar Boolean isRddPersistent: True if training RDD should be persistent
    :ivar Boolean isTest: true if we perform model test
    :ivar Boolean isTraining: true if we perform model training
    :ivar String label: Get/Set label blob name
    :ivar String labelFile: Get/Set the input label file
    :ivar int lmdb_partitions: Get/Set number of LMDB partitions
    :ivar log: Logger
    :ivar String modelPath: Get/Set HDFS path for model file
    :ivar NetParameter netParam: Get Network parameter
    :ivar String outputFormat: Get/Set output dataframe format.
    :ivar String outputPath: Get/Set HDFS path for output file for test results etc
    :ivar String protoFile: Get/Set file location of solver protocol file
    :ivar Boolean resize: True if input image will be resized
    :ivar String snapshotModelFile: Get/Set HDFS path for snapshot models
    :ivar String snapshotStateFile: Get/Set HDFS path for snapshot states
    :ivar SolverParameter solverParameter: Get Solver parameter
    :ivar int test_data_layer_id: Get layer ID of training data source
    :ivar int train_data_layer_id: Get layer ID of training data source
    :ivar int transform_thread_per_device: Get/Set # of transformer threads per device
    :ivar String imageCaptionDFDir: Path to generate the image caption dataframe
    :ivar String vocabDir: Path to generate the Vocab
    :ivar String embeddingDFDir: Path to generate the embedded dataframe
    :ivar String captionFile: Path to the caption file
    :ivar int captionLength: Embedding caption length
    :ivar int vocabSize: Vocab size to consider
    """
    def __init__(self,sc,args=None):
        registerContext(sc)
        if (args != None) :
            args_list = self.__convert_dict_to_list(args)
            self.__dict__['config']=wrapClass("com.yahoo.ml.caffe.Config")(sc,args_list)
        else :
            self.__dict__['config']=wrapClass("com.yahoo.ml.caffe.Config")(sc)
        
        
    def __convert_dict_to_list(self,conf_dict):
        conf_list=[]
        for key, value in conf_dict.iteritems():
            if (key[0] != '-'):
                key = '-' + key
            conf_list.append(key)
            conf_list.append(value)
        return conf_list

    def __setattr__(self, name, value):
        try:
            getattr(self.__dict__['config'], name+"_$eq")(value)
        except: 
            print "Attribute "+name+ " not valid"

    def __getattr__(self, name):
        try:
            return getattr(self.__dict__['config'], name)()
        except:
            print "Attribute "+name+ " not valid"
    
