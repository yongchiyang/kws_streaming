# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model based on combination of 2D depthwise and 1x1 convolutions."""
from kws_streaming.layers import modes
from kws_streaming.layers import speech_features
from kws_streaming.layers import stream
from kws_streaming.layers.compat import tf
import kws_streaming.models.model_utils as utils


def model_parameters(parser_nn):
  """Depthwise Convolutional(DS CNN) model parameters.

  In more details parameters are described at:
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/DepthwiseConv2D
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D

  Args:
    parser_nn: global command line args parser
  Returns: parser with updated arguments
  """
  parser_nn.add_argument(
     '--maxpool_size',
     type=int,
     default=1,
     help='Pooling size in the first layer',
  )
  
  parser_nn.add_argument(
     '--maxpool_stride',
     type=int,
     default=1,
     help='Pooling stride in the first layer',
  )
  
  parser_nn.add_argument(
      '--cnn1_kernel_size',
      type = int,
      default = 1,
      help='Heights and widths of the first 2D convolution',
  )
  
  parser_nn.add_argument(
      '--cnn1_strides',
      type=int,
      default=1,
      help='Strides of the first 1D convolution along the height and width',
  )
  parser_nn.add_argument(
      '--cnn1_padding',
      type=str,
      default='valid',
      help="One of 'valid' or 'same'",
  )
  parser_nn.add_argument(
      '--cnn1_filters',
      type=int,
      default=3,
      help='Number of output filters in the first 2D convolution layers',
  )
  
  parser_nn.add_argument(
      '--bn_momentum',
      type=float,
      default=0.1,
      help='Momentum for bn',
  )
  
  parser_nn.add_argument(
      '--bn_epsilon',
      type=float,
      default=1e-5,
      help='epsilon for bn',
  )
  
  parser_nn.add_argument(
      '--act',
      type=str,
      default="'relu','relu'",
      help='Activation function in the first 2D convolution layers',
  )
  
  parser_nn.add_argument(
      '--dw2_kernel_size',
      type=str,
      default='(1,1),(1,1)',
      help='kernel size of dw2',
  )
  parser_nn.add_argument(
      '--dw2_depth_multiplier',
      type=str,
      default='1,2',
      help='depth_multiplier of the 2D Depthwise convolutions',
  )
  parser_nn.add_argument(
      '--dw2_strides',
      type=str,
      default='(1,1),(1,1)',
      help='Strides of the 2D Depthwise convolutions',
  )
  parser_nn.add_argument(
      '--dw2_padding',
      type=str,
      default='valid',
      help="One of 'valid' or 'same'",
  )
  
  parser_nn.add_argument(
     '--avgpool_size',
     type=str,
     default='1,1',
     help='Pooling size in the middle layer',
  )
  
  parser_nn.add_argument(
     '--avgpool_stride',
     type=str,
     default='1,1',
     help='Pooling stride in the middle layer',
  )
  parser_nn.add_argument(
      '--dw3_kernel_size',
      type=str,
      default='(1,1),(1,1)',
      help='kernel size of dw3',
  )
  parser_nn.add_argument(
      '--dw3_depth_multiplier',
      type=str,
      default='2,2',
      help='depth_multiplier of the dw3',
  )
  parser_nn.add_argument(
      '--dw3_strides',
      type=str,
      default='(1,1),(1,1)',
      help='Strides of the dw3',
  )
  parser_nn.add_argument(
      '--dw3_padding',
      type=str,
      default='valid',
      help="One of 'valid' or 'same'",
  )
  parser_nn.add_argument(
      '--dropout',
      type=float,
      default=0.5,
      help='Percentage of data dropped',
  )
  parser_nn.add_argument(
      '--units1',
      type=int,
      default=10,
      help='dense layer 1',
  )
  parser_nn.add_argument(
      '--units2',
      type=int,
      default=2,
      help='dense layer 2',
  )
  
  
def model(flags):
  """Depthwise convolutional model.

  It is based on paper:
  MobileNets: Efficient Convolutional Neural Networks for
  Mobile Vision Applications https://arxiv.org/abs/1704.04861
  Model topology is similar with "Hello Edge: Keyword Spotting on
  Microcontrollers" https://arxiv.org/pdf/1711.07128.pdf
  Args:
    flags: data/model parameters

  Returns:
    Keras model for training
  """

  input_audio = tf.keras.layers.Input(
      shape=modes.get_input_data_shape(flags, modes.Modes.TRAINING),
      batch_size=flags.batch_size)
  net = input_audio

  if flags.preprocess == 'raw':
    # it is a self contained model, user need to feed raw audio only
    net = speech_features.SpeechFeatures(
        speech_features.SpeechFeatures.get_params(flags))(
            net)

  net = tf.keras.backend.expand_dims(net)
  
  print(net[0])
  
  net = tf.keras.layers.MaxPooling1D(
    pool_size = utils.parse(flags.maxpool_size),
    strides = utils.parse(flags.maxpool_stride))(net)
            
  net = stream.Stream(
        cell=tf.keras.layers.Conv1D(
            kernel_size=utils.parse(flags.cnn1_kernel_size),
            filters=flags.cnn1_filters,
            padding=flags.cnn1_padding,
            strides=utils.parse(flags.cnn1_strides)))(net)
            
  net = tf.keras.layers.BatchNormalization(
            momentum=flags.bn_momentum,
            epsilon=flags.bn_epsilon)(net)
            
  net = tf.keras.layers.Activation('relu')(net)
  
  for kernel_size, depth_mul, act, strides in zip(
      utils.parse(flags.dw2_kernel_size),
      utils.parse(flags.dw2_depth_multiplier),
      utils.parse(flags.act),
      utils.parse(flags.dw2_strides)):
    net = stream.Stream(
          cell=tf.keras.layers.DepthwiseConv2D(
          kernel_size=kernel_size,
          depth_multiplier=depth_mul,
          padding=flags.dw2_padding,
          strides=strides),
          use_one_step = False)(net)
    net = tf.keras.layers.BatchNormalization(
          momentum=flags.bn_momentum,
          epsilon=flags.bn_epsilon)(net)
    net = tf.keras.layers.Activation('relu')(net)
    
  net = stream.Stream(
        cell = tf.keras.layers.AveragePooling2D(
            pool_size = utils.parse(flags.avgpool_size),
            strides = utils.parse(flags.avgpool_stride)))(net)
            
  for kernel_size, depth_mul, act, strides in zip(
      utils.parse(flags.dw3_kernel_size),
      utils.parse(flags.dw3_depth_multiplier),
      utils.parse(flags.act),
      utils.parse(flags.dw3_strides)):
    net = stream.Stream(
          cell=tf.keras.layers.DepthwiseConv2D(
          kernel_size=kernel_size,
          depth_multiplier=depth_mul,
          padding=flags.dw3_padding,
          strides=strides),
          use_one_step = False)(net)
    net = tf.keras.layers.BatchNormalization(
          momentum=flags.bn_momentum,
          epsilon=flags.bn_epsilon)(net)
    net = tf.keras.layers.Activation('relu')(net)
    
  net = tf.keras.layers.Dropout(rate=flags.dropout)(net)
  net = stream.Stream(cell=tf.keras.layers.Flatten())(net)
  net = tf.keras.layers.Dense(units=flags.units1, activation='relu')(net)
  if flags.tinyml_test:
    net = tf.keras.layers.Dense(units=flags.units2, activation='softmax')(net)
  else:
    net = tf.keras.layers.Dense(units=12, activation='softmax')(net)

  return tf.keras.Model(input_audio, net)
