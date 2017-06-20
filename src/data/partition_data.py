"""
Partitions the data (images) located in data_dir into train and 
validation directories train_dir and validation_dir respectively

Partitions the data according to val_proportion argument which 
specifies what proportion of the data should be reserved to validation
"""
import numpy as np
from tqdm import tqdm
import os
import re
import util as u
import shutil
import tensorflow as tf

tf.app.flags.DEFINE_string(
    'validation_dir', '../../data/partitioned/validation/',
    'Directory containing the validation (images) dataset')

tf.app.flags.DEFINE_string(
    'train_dir', '../../data/partitioned/train/',
    'Directory containing the training (images) dataset')

tf.app.flags.DEFINE_string(
    'data_dir', '../../data/video18/',
    'Directory containing the whole image dataset')

tf.app.flags.DEFINE_string(
    'data_file', '../../data/data.json',
    'Name (including path) of the file containing the urls+labels')

tf.app.flags.DEFINE_float(
    'val_proportion', 0.2,
    'Proportion of dataset to be assigned to validation.'
    'should be between 0.0 and 1.0')


def main(_):
  # get flags
  FLAGS = tf.app.flags.FLAGS
  VAL_PROP = FLAGS.val_proportion
  DATA_DIR = FLAGS.data_dir
  VALID_DIR = FLAGS.validation_dir
  TRAIN_DIR = FLAGS.train_dir
  # for reprducibility
  RANDOM_SEED = 12345
  
  # delete files in train and validation directories
  u.delete_files_dir(VALID_DIR)
  u.delete_files_dir(TRAIN_DIR)
  
  # get list of file names in specified data directory
  fnames = u.get_fnames_in_dir(DATA_DIR)
  num_files = len(fnames)

  # shuffle the files and make the randomness repeatable
  shuffle_indices = list(range(num_files))
  np.random.seed(RANDOM_SEED)
  np.random.shuffle(shuffle_indices)
  fnames = [fnames[i] for i in shuffle_indices]

  slice_index = int(num_files * VAL_PROP)

  # copy to validation directory
  for fn in tqdm(fnames[:slice_index]):
    shutil.copy2(DATA_DIR+fn, VALID_DIR+fn)

  # copy to train directory
  for fn in tqdm(fnames[slice_index:]):
    shutil.copy2(DATA_DIR+fn, TRAIN_DIR+fn)

if __name__ == '__main__':
  tf.app.run()
