# Copyright 2018 The Authors Sunyaqiang. All Rights Reserved.
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
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
try:
    from bearing.deep_learning.models_mse_loss.read_util import tensorflow_input
except:
    from ..read_util import tensorflow_input
FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
# tf.app.flags.DEFINE_integer('batch_size', 40,
#                             """Number of images to process in a batch.""")


# tf.app.flags.DEFINE_string('data_dir', '/Volumes/SUNXL-FAT/Deep_Learning/chuangxin/chuangxin_Foxcoon/data/crop/',
#                            """Path to the CIFAR-10 data directory.""")
# 0402_1000

# path = r'G:\Bearing\bearing\data_set\FEMTOBearingDataSet\Training_set\Learning_set'

# path_test = path

# tf.app.flags.DEFINE_string('data_dir', train_path,
#                            """Path to the CIFAR-10 data directory.""")
# tf.app.flags.DEFINE_string('data_dir_test',path_test,
#                            """Path to the CIFAR-10 data directory.""")
# tf.app.flags.DEFINE_string('trainable_scopes', None,
#                            """Path to the CIFAR-10 data directory.""")

# tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', None,
#                            """Path to the CIFAR-10 data directory.""")

tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.

NUM_JUDGE = tensorflow_input.NUM_JUDGE
NUM_DEGREES = tensorflow_input.NUM_DEGREES

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = tensorflow_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = tensorflow_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
# MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.



# data for train
def distorted_inputs(batch_size,train_path_list,num_sets=1,input_dim=10):
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  # if not FLAGS.data_dir:
  #   raise ValueError('Please supply a data_dir')

  images_set = []
  labels_set = []

  data_dir = train_path_list
  images, labels1 = tensorflow_input.distorted_inputs(data_dir=data_dir,
                                                      batch_size=batch_size,input_dim=input_dim)
  # Randomly crop a [height, width] section of the image.
  # distorted_image = tf.random_crop(reshaped_image, [height, width,1])
  # res
  images.set_shape([batch_size, input_dim, 1])

  # tf.summary.image('images_random_croped', images)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels1 = tf.cast(labels1, tf.float16)
  for i in range(num_sets):
      images_set.append(images)
      labels_set.append(labels1)
  return images_set, labels_set



# data for test
def test_inputs(batch_size,test_path_list,num_sets=1,input_dim=10):
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  # if not FLAGS.data_dir:
  #   raise ValueError('Please supply a data_dir')
  images_set = []
  labels_set = []
  data_dir = test_path_list
  images, labels1 = tensorflow_input.test_inputs(data_dir=data_dir, batch_size=batch_size,input_dim=input_dim)
  images.set_shape([batch_size, input_dim, 1])
  # tf.summary.image('images_random_croped', images)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels1 = tf.cast(labels1, tf.float16)
  for i in range(num_sets):
      images_set.append(images)
      labels_set.append(labels1)
  return images_set, labels_set

if __name__=="__main__":
    distorted_inputs(64)