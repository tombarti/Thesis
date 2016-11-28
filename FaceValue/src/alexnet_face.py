import tensorflow.contrib.slim as slim
import tensorflow as tf


def alexnet(inputs):
  with slim.arg_scope([slim.ops.conv2d], weight_decay=0.0005,
                                         activation_fn=None):
  with arg_scope([slim.ops.batch_norm], activation_fn=tf.nn.relu):
    net = slim.ops.conv2d(inputs, 96, [11, 11], stride=4, scope='conv1')

  return net
