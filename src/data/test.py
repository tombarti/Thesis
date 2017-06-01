from PIL import Image
import numpy as np
import tensorflow as tf

def read_and_decode(filename_queue):
 reader = tf.TFRecordReader()
 _, serialized_example = reader.read(filename_queue)
 features = tf.parse_single_example(
  serialized_example,
  # Defaults are not specified since both keys are required.
  features={
      'image/encoded': tf.FixedLenFeature([], tf.string),
      'image/class/label': tf.FixedLenFeature([60], tf.int64),
      'image/height': tf.FixedLenFeature([], tf.int64),
      'image/width': tf.FixedLenFeature([], tf.int64)
  })
 image = tf.decode_raw(features['image/encoded'], tf.uint8)
 label = tf.cast(features['image/class/label'], tf.int32)
 height = tf.cast(features['image/height'], tf.int32)
 width = tf.cast(features['image/width'], tf.int32)
 return image, label, height, width


def get_all_records(FILE):
 with tf.Session() as sess:
   filename_queue = tf.train.string_input_producer([ FILE ])
   image, label, height, width = read_and_decode(filename_queue)
   init_op = tf.global_variables_initializer()
   sess.run(init_op)
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(coord=coord)
   for i in range(5):
     example, l, h, w= sess.run([image, label, height, width])
     print("Image size: {} x {}".format(h, w))

     print (example,l)
   coord.request_stop()
   coord.join(threads)

record_file = '../../data/records/validation-00000-of-00002'
get_all_records(record_file)
