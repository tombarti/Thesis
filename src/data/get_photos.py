# system imports
from functools import partial
import multiprocessing
import os
# third party imports
import tensorflow as tf
from tqdm import tqdm
import numpy as np
# local imports
import util as u


tf.app.flags.DEFINE_string(
    'download_dir', '../../data/imgs/',
    'Directory to which images will be downloaded')

tf.app.flags.DEFINE_string(
    'data_file', '../../data/EmotioNet_FACS.csv',
    'path+name of file containing list of urls and labels')

tf.app.flags.DEFINE_string(
    'err_file', './download_errs.txt',
    'path+name of file containing list of bad urls')

tf.app.flags.DEFINE_bool(
    'parallel', True,
    'Download images in parallel using multiprocessing')

def get_urls(data_file):
  """Returns a list of urls to download from given data file

  Args:
    data_file: string, path+filename to the file containing list of urls
  Returns:
    img_urls: [string], list of img urls to download
  """
  img_urls = []
  with open(data_file, 'r') as fh:
    lines = fh.readlines()[1:]
    img_urls = [l.split(',')[0][1:-1] for l in lines]
  # delete leading ' if there is one
  for i, img_url in enumerate(img_urls):
    if img_url[0] == "'":
      img_urls[i] = img_url[1:]
      
  img_urls = np.unique(img_urls)
  print("Found {} unique urls".format(len(img_urls)))
  return img_urls

def remove_already_downloaded_urls(urls, data_dir):
  """Removes urls that have already been downloaded from list

  Args: 
    urls: [string], the initial list of urls
    data_dir: string, path+name to directory in which images are downloaded
  Returns:
    not_downloaded: [string], list of urls that have not been downloaded
  """
  downloaded = u.get_fnames_in_dir(data_dir)
  downloaded = [u.del_file_ext(fn) for fn in downloaded]
  urls = [(url, u.del_file_ext(u.url_to_filename(url))) for url in urls]
  not_downloaded = [url for url, fn in urls if fn not in downloaded]
  print("Removed {} urls that were already downloaded".format(
    len(urls) - len(not_downloaded)))
  return not_downloaded

def remove_bad_urls(urls, err_file):
  """Removes urls that produce an error when downloading

  Args: 
    urls: [string], the initial list of urls
    err_file: string, path+name to file of erroneous urls
  Returns:
    clean_urls: [string], list of urls that do not have errors
  """
  bad_urls = []
  if not os.path.isfile(err_file):
    print("No download errors file present, removed 0 bad urls")
    return urls
  with open(err_file, 'r') as fh:
    bad_urls = [l.split(',')[0] for l in fh.readlines()]
  clean_urls = []
  for url in urls:
    if not url in bad_urls:
      clean_urls.append(url)
  print("Removed {} bad urls".format(len(urls) - len(clean_urls)))
  return clean_urls

def clean_error_file(err_file):
  lines = []
  delete_indexes = []
  with open(err_file, 'r') as fh:
    lines = fh.readlines()
  errors = [l.split(',') for l in lines]
  n_errs = len(lines)
  # go through errors to find those corresponding to service not known
  for i,l in enumerate(errors):
    if len(l) != 2:
      delete_indexes.append(i)
      continue
    err = l[1].strip()
    print(err)
    if err == "<urlopen error unknown url type: 'http>":
      delete_indexes.append(i)
    elif err == "<urlopen error [Errno -2] Name or service not known>":
      delete_indexes.append(i)

  # know delete false errors
  for del_ind in sorted(delete_indexes, reverse=True):
    del lines[del_ind]
  # remove old error file
  #os.remove(err_file)
  # write new error file
  #with open(err_file, 'w') as fh:
    #fh.writelines(lines)
  #print('Removed {} errors from {}'.format(n_errs - len(lines), err_file))
  


def main(_):
  # get flags
  FLAGS = tf.app.flags.FLAGS
  DOWNLOAD_DIR = FLAGS.download_dir
  DATA_FILE = FLAGS.data_file

  # get list of all urls
  img_urls = get_urls(data_file=DATA_FILE)
  # remove urls already downloaded
  img_urls = remove_already_downloaded_urls(img_urls, DOWNLOAD_DIR)
  # remove connection errors
  #clean_error_file(FLAGS.err_file)
  # remove bad urls
  img_urls = remove_bad_urls(img_urls, FLAGS.err_file)
  print("Number of images to download: {}".format(len(img_urls)))

  if FLAGS.parallel:
    print("Downloading images in parallel")
    # create partial download function for mapping
    partial_download = partial(u.download_img_from_url, 
        data_dir=DOWNLOAD_DIR)
    # create pool for multiprocessing
    pool = multiprocessing.Pool(processes=8)
    results = pool.map(partial_download, img_urls)
    # close all processes
    pool.close()
    # join results
    pool.join()
  else:
    print("Downloading images sequentially")
    for url in tqdm(img_urls):
      u.download_img_from_url(url, DOWNLOAD_DIR)

if __name__ == '__main__':
  tf.app.run()
  
