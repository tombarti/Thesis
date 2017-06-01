# system imports
import shutil
import pickle
import os
# third party imports
import tensorflow as tf
import numpy as np
from tqdm import tqdm
# local imports


tf.app.flags.DEFINE_string(
    'images_dir', './data/images/',
    'Directory containing the images to annotate')

tf.app.flags.DEFINE_string(
    'destination_dir', './valence_arousal_annotator/annotator/static/data/',
    'Directory to which the processed data should be saved')

tf.app.flags.DEFINE_integer(
    'video_size', 1000,
    'Number of frames in a single video')

def main(_):
  FLAGS = tf.app.flags.FLAGS
  # will contain the mapping between frame numbers and filenames
  img_dict = {}
  # get list of images (filenames)
  filenames = os.listdir(FLAGS.images_dir)
  n_files = len(filenames)
  # compute number of video subfolders to create
  n_videos = int(np.floor(n_files / FLAGS.video_size))
  # save N = FLAGS.video_size image as frames into each video subfolder
  for i in range(n_videos):
    video_dir = FLAGS.destination_dir + 'video{}'.format(i+1)
    #os.mkdir(video_dir)
    print("Filling video {} subfolder with frames". format(i+1))
    for j in tqdm(range(FLAGS.video_size)):
      fname = filenames.pop()
      frame = '{:04}.png'.format(j+1)
      img_dict[frame] = fname
      #shutil.copy2(FLAGS.images_dir + fname, video_dir + frame)

  # if there are remaining files, copy  them to another videoN directory
  if len(filenames) != 0:
    video_dir = FLAGS.destination_dir + 'video{}'.format(n_videos+1)
    #os.mkdir(video_dir)
    print("Filling video {} subfolder with remainig {} frames"
        .format(n_videos+1, len(filenames)))
    for k, fname in enumerate(filenames):
      frame = '{:04}.png'.format(k+1)
      img_dict[frame] = fname
      #shutil.copy2(FLAGS.images_dir + fname, video_dir + frame)



if __name__ == '__main__':
  tf.app.run()
