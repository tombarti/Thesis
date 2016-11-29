import scipy.io as sio
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim



def mat2tf(fname, name='mat2tf_translation'):
  net_desc = sio.loadmat(fname)
  layers = net_desc['layers'][0]
  trainOpts = net_desc['meta']['trainOpts'][0][0]
  # get shape of input image
  in_shape = extract_arr(net_desc['meta']['inputSize'])
  in_shape = tuple(np.insert(in_shape,0,1))
  print(in_shape)
  inputs = tf.placeholder(tf.float32, in_shape)
  net = build_net(inputs, layers)

def PADDING(n):
  if n == 0:
    return 'VALID'
  else:
    return 'SAME'

def build_conv2d(net, layer_desc):
  print('Layer: conv2d')
  scope = extract_val(layer_desc['name'])
  weights = extract_arr(layer_desc['weights'])
  b_shape = weights[2].shape
  print(b_shape)
  stride = extract_val(layer_desc['stride'])
  stride = [1, stride, stride, 1]
  pad = extract_val(layer_desc['pad'])
  print('  - scope  : {}'.format(scope))
  print('  - weights: {}'.format(weights[0].shape))
  print('  - biases : {}'.format(weights[1].shape))
  print('  - stride : {}'.format(stride))
  print('  - pading : {}'.format(pad))
  with tf.name_scope(scope) as scope:
    kernel = tf.Variable(weights[0], name='weights')
    net = tf.nn.conv2d(net, kernel, stride, padding=PADDING(pad))
    biases = tf.Variable(weights[1], trainable=True, name='biases')
    net = tf.nn.bias_add(net, biases)
  return net

  

def build_batch_norm(net, layer_desc):
  print('Layer: batch_norm')
  weights = extract_arr(layer_desc['weights'])
  print('  - scope  : {}'.format(extract_val(layer_desc['name'])))
  print('  - weights: {}'.format(weights.shape))
  print('  - weight : {}'.format(weights[0].shape))

def build_relu(net, layer_desc):
  print('Layer: relu')
  print('  - scope  : {}'.format(extract_val(layer_desc['name'])))
  print('  - leak   : {}'.format(extract_val(layer_desc['leak'])))

def build_pool(net, layer_desc):
  print('Layer: pool')
  print('  - scope  : {}'.format(extract_val(layer_desc['name'])))
  print('  - method : {}'.format(extract_val(layer_desc['method'])))
  print('  - pading : {}'.format(extract_val(layer_desc['pad'])))
  print('  - stride : {}'.format(extract_val(layer_desc['stride'])))
  print('  - kernel : {}'.format(extract_arr(layer_desc['pool'])))

def build_softmaxloss(net, layer_desc):
  print('Layer: softmaxloss')

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
    net = layer_builder[l_type](net, l)

def extract_val(arr):
  while isinstance(arr, (np.ndarray)):
    arr = arr[0]
  return arr

def extract_arr(arr):
  while arr.shape == (1,1) or arr.shape == (1,):
    arr = arr[0]
  # get rid of redundant index ...
  if len(arr.shape) == 2:
    if arr.shape[0] == 1:
      arr = arr.reshape((arr.shape[1],))
  return arr



if __name__ == '__main__':
  mat2tf('/data/teb13/nets/matlab/alexnet-face-bn.mat')
