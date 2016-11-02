#!/usr/bin/env python
# This file is a modified version of: https://github.com/jeffdonahue/caffe/blob/506c4d94179afab4eeb5b0da39b65239e40e25fb/examples/coco_caption/retrieval_experiment.py
from collections import OrderedDict
import json
import numpy as np
import pprint
import cPickle as pickle
import string
import sys
from pyspark.sql import SQLContext
from pyspark import SparkConf,SparkContext
from pyspark.sql import Row
from captioner import Captioner
import array
import io
from PIL import Image
import numpy as np

class CaptionExperiment():
  def __init__(self, image_model, image_net_proto, lstm_net_proto, vocab):
    self.captioner = Captioner(image_model, image_net_proto, lstm_net_proto, vocab,-1)
    self.captioner.set_image_batch_size(1)
  

  def getCaption(self, image):
    row=image
    bytes = array.array('b', row.image).tostring()
    im = Image.open(io.BytesIO(bytes))
    image = np.array(im,dtype=np.uint8)
    dataset = [image]
    descriptors = self.captioner.compute_descriptors(dataset)
    images = dataset
    num_images = len(images)
    batch_size = num_images
  
    #Generate captions for all images.
    all_captions = [None] * num_images
    for image_index in xrange(0, num_images, batch_size):
      batch_end_index = min(image_index + batch_size, num_images)
      output_captions, output_probs = self.captioner.sample_captions(
        descriptors[image_index:batch_end_index], temp=float('inf'))
      for batch_index, output in zip(range(image_index, batch_end_index),
                                     output_captions):
        all_captions[batch_index] = output
  #
  #    # Collect model/reference captions, formatting the model's captions and
  #    # each set of reference captions as a list of len(self.images) strings.
  #    # For each image, write out the highest probability caption.
      model_captions = [''] * len(images)
      for image_index, image in enumerate(images):
        caption = self.captioner.sentence(all_captions[image_index])
        model_captions[image_index] = caption

      generation_result = [Row(row.id,model_captions[image_index]) for (image_index, image_path) in enumerate(images)]      
      return generation_result

