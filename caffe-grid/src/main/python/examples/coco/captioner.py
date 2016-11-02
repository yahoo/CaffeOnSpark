#!/usr/bin/env python
# This file is a modified version of: https://github.com/jeffdonahue/caffe/blob/506c4d94179afab4eeb5b0da39b65239e40e25fb/examples/coco_caption/captioner.py
from collections import OrderedDict
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
from PIL import Image
import caffe

class Captioner():
  def __init__(self, weights_path, image_net_proto, lstm_net_proto,
               vocab_path, device_id=-1):
    if device_id >= 0:
      caffe.set_mode_gpu()
      caffe.set_device(device_id)
    else:
      caffe.set_mode_cpu()
    # Setup image processing net.
    phase = caffe.TEST
    print 'weights_path:' + weights_path
    print 'image_net_proto:' + image_net_proto
    print 'lstm_net_proto:' + lstm_net_proto
    self.image_net = caffe.Net(image_net_proto, weights_path, phase)
    image_data_shape = self.image_net.blobs['data'].data.shape
    self.transformer = caffe.io.Transformer({'data': image_data_shape})
    channel_mean = np.zeros(image_data_shape[1:])
    channel_mean_values = [104, 117, 123]
    assert channel_mean.shape[0] == len(channel_mean_values)
    for channel_index, mean_val in enumerate(channel_mean_values):
      channel_mean[channel_index, ...] = mean_val
    self.transformer.set_mean('data', channel_mean)
    self.transformer.set_channel_swap('data', (2, 1, 0))
    self.transformer.set_transpose('data', (2, 0, 1))
    # Setup sentence prediction net.
    self.lstm_net = caffe.Net(lstm_net_proto, weights_path, phase)
    self.vocab = ['<EOS>']
    with open(vocab_path, 'r') as vocab_file:
      self.vocab += [word.strip() for word in vocab_file.readlines()]
    net_vocab_size = self.lstm_net.blobs['predict'].data.shape[2]
    if len(self.vocab) != net_vocab_size:
      raise Exception('Invalid vocab file: contains %d words; '
          'net expects vocab with %d words' % (len(self.vocab), net_vocab_size))

  def set_image_batch_size(self, batch_size):
    self.image_net.blobs['data'].reshape(batch_size,
        *self.image_net.blobs['data'].data.shape[1:])

  def caption_batch_size(self):
    return self.lstm_net.blobs['cont_sentence'].data.shape[1]

  def set_caption_batch_size(self, batch_size):
    self.lstm_net.blobs['cont_sentence'].reshape(1, batch_size)
    self.lstm_net.blobs['input_sentence'].reshape(1, batch_size)
    self.lstm_net.blobs['image_features'].reshape(batch_size,
        *self.lstm_net.blobs['image_features'].data.shape[1:])
    self.lstm_net.reshape()

  def preprocess_image(self, image, verbose=False):
    if type(image) in (str, unicode):
      image = plt.imread(image)
    
    crop_edge_ratio = (256. - 227.) / 256. / 2
    ch = int(image.shape[0] * crop_edge_ratio + 0.5)
    cw = int(image.shape[1] * crop_edge_ratio + 0.5)
    cropped_image = image[ch:-ch, cw:-cw]
    if len(cropped_image.shape) == 2:
      cropped_image = np.tile(cropped_image[:, :, np.newaxis], (1, 1, 3))
    preprocessed_image = self.transformer.preprocess('data', cropped_image)
    if verbose:
      print 'Preprocessed image has shape %s, range (%f, %f)' % \
          (preprocessed_image.shape,
           preprocessed_image.min(),
           preprocessed_image.max())
    return preprocessed_image


  def compute_descriptors(self, image_list, output_name='fc8'):
    batch = np.zeros_like(self.image_net.blobs['data'].data)
    batch_shape = batch.shape
    batch_size = batch_shape[0]
    descriptors_shape = (len(image_list), ) + \
        self.image_net.blobs[output_name].data.shape[1:]
    descriptors = np.zeros(descriptors_shape)
    for batch_start_index in range(0, len(image_list), batch_size):
      batch_list = image_list[batch_start_index:(batch_start_index + batch_size)]
      for batch_index, image_path in enumerate(batch_list):
        batch[batch_index:(batch_index + 1)] = self.preprocess_image(image_path)
      current_batch_size = min(batch_size, len(image_list) - batch_start_index)
      self.image_net.forward(data=batch)
      descriptors[batch_start_index:(batch_start_index + current_batch_size)] = \
          self.image_net.blobs[output_name].data[:current_batch_size]
    return descriptors


  def sample_captions(self, descriptor, prob_output_name='probs',
                      pred_output_name='predict', temp=1, max_length=50):
    descriptor = np.array(descriptor)
    batch_size = descriptor.shape[0]
    self.set_caption_batch_size(batch_size)
    net = self.lstm_net
    cont_input = np.zeros_like(net.blobs['cont_sentence'].data)
    word_input = np.zeros_like(net.blobs['input_sentence'].data)
    image_features = np.zeros_like(net.blobs['image_features'].data)
    image_features[:] = descriptor
    outputs = []
    output_captions = [[] for b in range(batch_size)]
    output_probs = [[] for b in range(batch_size)]
    caption_index = 0
    num_done = 0
    while num_done < batch_size and caption_index < max_length:
      if caption_index == 0:
        cont_input[:] = 0
      elif caption_index == 1:
        cont_input[:] = 1
      if caption_index == 0:
        word_input[:] = 0
      else:
        for index in range(batch_size):
          word_input[0, index] = \
              output_captions[index][caption_index - 1] if \
              caption_index <= len(output_captions[index]) else 0
      net.forward(image_features=image_features, cont_sentence=cont_input,
                  input_sentence=word_input)
      if temp == 1.0 or temp == float('inf'):
        net_output_probs = net.blobs[prob_output_name].data[0]
        samples = [
            random_choice_from_probs(dist, temp=temp, already_softmaxed=True)
            for dist in net_output_probs
        ]
      else:
        net_output_preds = net.blobs[pred_output_name].data[0]
        samples = [
            random_choice_from_probs(preds, temp=temp, already_softmaxed=False)
            for preds in net_output_preds
        ]
      for index, next_word_sample in enumerate(samples):
        # If the caption is empty, or non-empty but the last word isn't EOS,
        # predict another word.
        if not output_captions[index] or output_captions[index][-1] != 0:
          output_captions[index].append(next_word_sample)
          output_probs[index].append(net_output_probs[index, next_word_sample])
          if next_word_sample == 0: num_done += 1
      caption_index += 1
    return output_captions, output_probs

  def sentence(self, vocab_indices):
    sentence = ' '.join([self.vocab[i] for i in vocab_indices])
    if not sentence: return sentence
    sentence = sentence[0].upper() + sentence[1:]
    # If sentence ends with ' <EOS>', remove and replace with '.'
    # Otherwise (doesn't end with '<EOS>' -- maybe was the max length?):
    # append '...'
    suffix = ' ' + self.vocab[0]
    if sentence.endswith(suffix):
      sentence = sentence[:-len(suffix)] + '.'
    else:
      sentence += '...'
    return sentence

def softmax(softmax_inputs, temp):
  shifted_inputs = softmax_inputs - softmax_inputs.max()
  exp_outputs = np.exp(temp * shifted_inputs)
  exp_outputs_sum = exp_outputs.sum()
  if math.isnan(exp_outputs_sum):
    return exp_outputs * float('nan')
  assert exp_outputs_sum > 0
  if math.isinf(exp_outputs_sum):
    return np.zeros_like(exp_outputs)
  eps_sum = 1e-20
  return exp_outputs / max(exp_outputs_sum, eps_sum)

def random_choice_from_probs(softmax_inputs, temp=1, already_softmaxed=False):
  # temperature of infinity == take the max
  if temp == float('inf'):
    return np.argmax(softmax_inputs)
  if already_softmaxed:
    probs = softmax_inputs
    assert temp == 1
  else:
    probs = softmax(softmax_inputs, temp)
  r = random.random()
  cum_sum = 0.
  for i, p in enumerate(probs):
    cum_sum += p
    if cum_sum >= r: return i
  return 1  # return UNK?

