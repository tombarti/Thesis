import multiprocessing
import pickle
import tensorflow as tf
import numpy as np
import util
from tqdm import tqdm
import os
try:
  import urllib.request as urllib2
except ImportError:
  import urllib2

DATA_DIR = "img/"

def get_urls(data_file='EmotioNet_FACS.csv'):
  img_urls = []
  with open(data_file, 'r') as fh:
    lines = fh.readlines()[1:]
    img_urls = [l.split(',')[0][1:-1] for l in lines]
    n_urls = len(img_urls)
  return remove_already_downloaded(img_urls)

def remove_already_downloaded(urls, data_dir='img/'):
  downloaded = util.get_fnames_in_dir(data_dir)
  res = []
  for url in urls:
    if not util.url_to_fname(url) in downloaded:
      res.append(url)
  return res

if __name__ == '__main__':
  DATA_FILE = "EmotioNet_FACS.csv"
  img_urls = get_urls(DATA_FILE)
  pool = multiprocessing.Pool(processes=6)
  results = pool.map(util.download_img_from_url, img_urls)
  pool.close()
  pool.join()
  
  bad_urls = [r for r in results if r[1] is not None]
  print(bad_urls)
  # save bad urls to file
  with open('bad_urls.pickle', 'wb') as fh:
    pickle.dump(bad_urls, fh)

