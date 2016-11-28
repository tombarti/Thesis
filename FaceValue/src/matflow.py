import scipy.io as sio
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim



def mat2tf(fname, name='mat2tf_translation'):
  net_desc = sio.loadmat(fname)
  layers = net_desc['layers'][0]
  trainOpts = net_desc['meta']['trainOpts'][0][0]
  # get shape of input image
  in_shape = tuple(net_desc['meta']['inputSize'][0][0][0])
  inputs = tf.placeholder(tf.int32, in_shape)
  net = build_net(inputs, layers)

def build_conv2d(layer_desc):
  print('building: conv2d')

def build_batch_norm(layer_desc):
  print('building: batch_norm')

def build_relu(layer_desc):
  print('building: relu')

def build_pool(layer_desc):
  print('building: pool')

def build_softmaxloss(layer_desc):
  print('building: softmaxloss')

# dictionnary of building functions
layer_builder = {
  'conv'       : build_conv2d,
  'bnorm'      : build_batch_norm,
  'relu'       : build_relu,
  'pool'       : build_pool,
  'softmaxloss': build_softmaxloss
}

def build_net(inputs, layers):
  print("build_net - {} layers found".format(len(layers)))
  net = inputs
  # build net layer by layer
  for i, l in enumerate(layers):
    l_type = extract_val(l['type'])
    # call approriate function to build this layer type
    net = layer_builder[l_type](l)

def extract_val(arr):
  while isinstance(arr, (np.ndarray)):
    arr = arr[0]
  return arr



if __name__ == '__main__':
  mat2tf('/data/teb13/nets/matlab/alexnet-face-bn.mat')
