from urllib.request import urlopen, Request
import urllib.error
import os

USER_AGENT = "Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11"

def download_img_from_url(url, data_dir='images/'):
  print(count_files_dir(data_dir))
  fname = url_to_fname(url)
  img, err = decode_jpeg_url(url)
  if not err:
    with open(data_dir + fname, 'wb') as fh:
      fh.write(img)
  return fname, err

def url_to_fname(url):
  fname = url.split('/')[-1]
  if len(fname) < 4:
    fname += '.jpg'
  elif fname[-4] == '.':
    fname = fname[0:-3] + 'jpg'
  else:
    fname += '.jpg'
  return fname

def count_files_dir(directory):
  """ Counts the number of files in a given directory """
  a = 0
  for _ in os.listdir(directory):
    a += 1
  return a

def decode_jpeg_url(url):
  img = None
  err = None
  try:
    img = urlopen(Request(url, headers={'User-Agent': USER_AGENT}))
    img = img.read()
  except Exception as e:
    err = e
    print("Error: [{}] for url: [{}]".format(err, url))
  return img, err

def get_fnames_in_dir(directory):
  """ Returns a list of the filenames in given directory"""
  return [f for f in os.listdir(directory)]

def rm_file_ext(fname):
  if not '.' in fname or len(fname) < 5:
    return fname
  elif fname[-4] == '.':
    return fname[:-4]






