import numpy as np
from tqdm import tqdm
import os
import re
import util
import shutil



def build_data_labels_dict():
  data = []
  with open('EmotioNet_FACS.csv', 'r') as fh:
    data = fh.readlines()[1:]

  data = [d.split(',') for d in data]
  for d in data:
    fname = util.url_to_fname(d[0][1:-1])
    d[0] = util.rm_file_ext(fname)
  data = [(d[0], d[1:]) for d in data]
  return dict(data)

def delete_files_dir(directory):
  fnames = [directory+f for f in os.listdir(directory)]
  for fn in fnames:
    os.remove(fn)
  print("removed {} files from {}".format(len(fnames), directory))


def main():
  RANDOM_SEED = 12345
  VAL_PROP = 0.3
  DATA_DIR = './images/'
  VALID_DIR = './organised_data/validation/'
  TRAIN_DIR = './organised_data/train/'
  
  # delete files in train and validation directories
  delete_files_dir(VALID_DIR)
  delete_files_dir(TRAIN_DIR)
  
  data_label_dict = build_data_labels_dict()
  fnames = util.get_fnames_in_dir(DATA_DIR)

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
  main()
