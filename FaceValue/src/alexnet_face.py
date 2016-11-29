import tensorflow.contrib.slim as slim
import tensorflow as tf

# each max pool layer has stride=2 and padding='VALID' by default

def alexnet(inputs):
  with slim.arg_scope([slim.conv2d], activation_fn=None, padding='VALID'):
    with slim.arg_scope([slim.batch_norm], activation_fn=tf.nn.relu):
      net = slim.conv2d(inputs, 96, [11, 11], stride=4, scope='conv1')
      net = slim.batch_norm(net, scope='bn1')
      net = slim.max_pool2d(net, [3, 3], scope='pool1')
      
      net = slim.conv2d(net, 256, [5, 5], stride=1, scope='conv2')
      net = slim.batch_norm(net, scope='bn2')
      net = slim.max_pool2d(net, [3, 3], scope='pool2')

      net = slim.conv2d(net, 384, [3, 3], stride=1, scope='conv3')
      net = slim.batch_norm(net, scope='bn3')

      net = slim.conv2d(net, 384, [3, 3], stride=1, scope='conv4')
      net = slim.batch_norm(net, scope='bn4')
      
      net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv5')
      net = slim.batch_norm(net, scope='bn5')

      net = slim.fully_connected(net, 4096, activation_fn=None, scope='fc6')
      net = slim.batch_norm(net, scope='bn6')

      net = slim.fully_connected(net, 4096, activation_fn=None, scope='fc7')
      net = slim.batch_norm(net, scope='bn7')

      net = slim.fully_connected(net, 2622, activation_fn=None, scope='fc8')

  return net


if __name__ == '__main__':
  inputs = tf.placeholder(tf.float32, (1, 227, 227, 3))
  net = alexnet(inputs)
