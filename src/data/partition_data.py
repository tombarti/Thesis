import numpy as np
from tqdm import tqdm
import os
import re
import util as u
import shutil
import tensorflow as tf

tf.app.flags.DEFINE_string(
    'validation_dir', '../../data/partitioned/validation',
    'Directory containing the validation (images) dataset')

tf.app.flags.DEFINE_string(
    'train_dir', '../../data/partitioned/train',
    'Directory containing the training (images) dataset')

tf.app.flags.DEFINE_string(
    'data_dir', '../../data/images/',
    'Directory containing the whole image dataset')

tf.app.flags.DEFINE_string(
    'data_file', '../../data/EmotioNet_FACS.csv',
    'Name (including path) of the file containing the urls+labels')

tf.app.flags.DEFINE_float(
    'val_proportion', 0.3,
    'Proportion of dataset to be assigned to validation.'
    'should be between 0.0 and 1.0')


def build_data_labels_dict():
  """Builds dictionary of filenames with their associated label

  Returns:
    data: dict, dictionary where key=filename and value=label
  """
  data = []
  # get urls & associated labels from csv file
  with open('EmotioNet_FACS.csv', 'r') as fh:
    data = fh.readlines()[1:]

  data = [d.split(',') for d in data]
  # replace urls with filenames (remove file extension as well)
  for d in data:
    # get the file name
    fname = u.url_to_filename(d[0][1:-1])
    # remove file extension
    d[0] = u.del_file_ext(fname)
  data = [(d[0], d[1:]) for d in data]
  return dict(data)

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

  # shuffle the files and make the randomness repeatable
  shuffle_indices = list(range(len(fnames)))
  np.random.seed(RANDOM_SEED)
  np.random.shuffle(shuffle_indices)
  fnames = [fnames[i] for i in shuffle_indices]

  slice_index = int(len(fnames) * VAL_PROP)

  # copy to validation directory
  for fn in tqdm(fnames[:slice_index]):
    shutil.copy2(DATA_DIR+fn, VALID_DIR+fn)

  # copy to train directory
  for fn in tqdm(fnames[slice_index:]):
    shutil.copy2(DATA_DIR+fn, TRAIN_DIR+fn)

if __name__ == '__main__':
  tf.app.run()
