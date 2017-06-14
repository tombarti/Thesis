# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_boolean(
    'is_multi_label', True,
    'If the classification task is mutlilabel ')

FLAGS = tf.app.flags.FLAGS

def _preprocess_labels(labels):
  """Replaces all 999 elements in labels by 0

  Args:
  labels: [batch_size, num_classes], the labels for n=batch_size examples

  Returns:
  labels: [batch_size, num_classes], the preprocessed labels with every 999
          replaced by a 0
  """
  BAD_LABEL = 999
  # iterate through each label
  for i, label in enumerate(labels):
    for j, l in enumerate(label):
      if l == BAD_LABEL:
        labels[i,j] = 0
  return labels

def _partial_accuracy(labels, predictions, num_classes):
  """Calculates the accuracy of predictions with respect to labels in a multi
  label setting (i.e. classes are not mutually exlusive: an example can have
  multiple classes active at the same time).

  The accuracy of one prediction (i.e. for one example) is calculated as the
  proportion of classes that where correctly predicted (e.g. the proportion of
  action units that were correctlly predicted).

  Args:
  labels: [batch_size, num_classes], tensor containing the labels for
          n=batch_size examples
  predictions: [batch_size, num_classes], tensor containing the predictions for
               n=batch_size examples

  Returns:
  partial_accuracies: [batch_size], the individual partial accuracies of each 
                      example 
  """
  partial_accuracies = []
  # make sure predictions and labels have the same type
  if labels.dtype != predictions.dtype:
    predictions = tf.cast(predictions, labels.dtype)
  
  # calculate partial accuracy of each prediction in predictions
  for label, prediction in zip(tf.unstack(labels), tf.unstack(predictions)):
    # compute number of correctly predicted classes
    matches = tf.reduce_sum(tf.cast(tf.equal(label, prediction), tf.float32))
    # divide by the number of classes to get proportion 
    partial_accuracies.append(matches / num_classes)
  return partial_accuracies

def clean_labels_predictions(labels, predictions):
  """Cleans the labels and their associated predicitons

  Here cleaning is defined as removing label elements that are equal to 999
  (i.e label elements that signify that we do not know if the action unit is
  activated or not)
  
  Hence we remove these "bad" elements from the label and remove the
  corresponding prediciton elements. This process is repeated for every label
  and prediction in arguments labels and predictions respectively.

  Args:
  labels: [batch_size, num_classes], tensor containing the labels for
          n=batch_size examples
  predictions: [batch_size, num_classes], tensor containing the predictions for
               n=batch_size examples

  Returns:
  clean_labels: [batch_size, num_classes], the cleaned labels
  clean_predictions: [batch_size, num_classes], the cleaned predictions
  """
  # make sure labels and predicitions have the same type
  if labels.dtype != predictions.dtype:
    predictions = tf.cast(predictions, labels.dtype)
  BAD_LABEL = tf.constant(999, dtype=tf.int64)
  clean_labels = []
  clean_predictions = []
  # clean each label and associated prediction
  for label, prediction in zip(tf.unstack(labels), tf.unstack(predictions)):
    # will be False where label = 999 and True otherwise
    delete_mask = tf.not_equal(label, BAD_LABEL)
    # gather label elements that are not equal to 999
    clean_labels.append(tf.boolean_mask(label, delete_mask))
    # gather associated predictions
    clean_predictions.append(tf.boolean_mask(prediction, delete_mask))
  return tf.stack(clean_labels), tf.stack(clean_predictions)

def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image, label] = provider.get(['image', 'label'])
    label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    # preprocess image
    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    # barch of images and corresponding labels
    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(images)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    if FLAGS.is_multi_label:
      print("Using multilabel approach\n")
      predictions = tf.round(tf.nn.sigmoid(logits))
    else:
      print("Usint multicalss approach\n")
      predictions = tf.argmax(logits, 1)

    # labels without the 999 elements and their corresponding predictions
    clean_labels, clean_predictions = clean_labels_predictions(labels,
                                                          predictions)
    # preprocess labels to replace all 999 labels with 0 
    l_shape = labels.get_shape()
    labels = tf.py_func(_preprocess_labels, [labels], tf.int64)
    labels.set_shape(l_shape)

    # remove uncessary outer dimensions
    labels = tf.squeeze(labels)

    # partial accuracy
    partial_accuracy = _partial_accuracy(labels, predictions, dataset.num_classes)

    # Define the metrics:
    # Partial accuracy is the average of the accuracy of each prediction.
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Partial_Accuracy': slim.metrics.streaming_mean(partial_accuracy),
        'Total_Accuracy' : slim.metrics.streaming_accuracy(predictions,
          labels),
        'Recall': slim.metrics.streaming_recall(
            logits, labels),
    })

    # Print the summaries to screen.
    for name, value in names_to_values.items():
      summary_name = 'eval/%s' % name
      op = tf.summary.scalar(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    slim.evaluation.evaluate_once(
        master=FLAGS.master,
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=list(names_to_updates.values()),
        variables_to_restore=variables_to_restore)


if __name__ == '__main__':
  tf.app.run()
