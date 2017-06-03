from urllib.request import urlopen, Request
import string
import pickle
from urllib.parse import urlparse
import urllib.error
import os
import imghdr

USER_AGENT = "Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11"

def download_img_from_url(url, data_dir, err_filename='download_errs.txt'):
  """Downloads image from url to file

  Args:
    url: string, the url of the image to download
    data_dir: string, the name of the dir on which to store img
    err_filename: string, name of filename in which to store errors
  Returns:
    fname: string, the file name to which image was stored
    err: Error, the error if any (None otherwise)
  """
  # decode the url to get bytes array and error
  img, err = decode_jpeg_url(url)
  # report error if there is one
  if err:
    with open(err_filename, 'a') as fh:
      fh.write('{},{}\n'.format(url, str(err)))
    return url, err
  else:
    print(count_files_dir(data_dir))
    img_type = imghdr.what('', h=img)
    fname = url_to_filename(url, ext=img_type)
    with open(data_dir + fname, 'wb') as fh:
      fh.write(img)
    return fname, err

def url_to_filename(url, ext=None):
  """Transforms an URL into a filename with appropriate extension
  
  Args:
    url: string, the url to be transformed
    ext: the file extension to use
  Returns:
    filename: string, the corresponding filename
  """
  _, old_ext = os.path.splitext(urlparse(url).path[1:])
  # remove punctuation from url
  filename = url.translate(str.maketrans('/', '_', '.'))
  # if an extension is specified, remove old one and add new
  if ext:
    filename, _ = os.path.splitext(filename)
    filename  += '.{}'.format(ext)
    
  # filenames can't be more than 255 chars
  if len(filename) > 254:
    limit = -254
    if not ext:
      limit += len(old_ext)
      if old_ext in ['.jpg', '.JPG']:
        limit += 1
    filename = filename[limit:]
  return filename

def del_file_ext(filename):
  filename, _ = os.path.splitext(filename)
  return filename


def count_files_dir(directory):
  """ Counts the number of files in a given directory 
  
  Args: 
    directory: string, the name of the directory including path
  Returns:
    count: int, the number of files in that directory
  """
  count = 0
  for _ in os.listdir(directory):
    count += 1
  return count

def delete_files_dir(directory):
  fnames = [directory+f for f in os.listdir(directory)]
  for fn in fnames:
    os.remove(fn)
  print("removed {} files from {}".format(len(fnames), directory))


def decode_jpeg_url(url):
  """Extracts the bytes stream of an image from its url

  Args:
    url: string, the url of the image to download
  Returns:
    img: [bytes], the bytes of the image
    err: Error, if an error was encounterd whilst downloading
  """
  img = None
  err = None
  # try to download image
  try:
    img = urlopen(Request(url, headers={'User-Agent': USER_AGENT}))
    img = img.read()
  except Exception as e:
    err = e
    #print("Error: [{}] for url: [{}]".format(err, url))
  return img, err

def get_fnames_in_dir(directory):
  """Returns a list of the filenames in given directory
  
  Args:
    directory: string, the path+name of the directory
  Returns:
    files: [string], list of files in the given direectory
  """
  return [f for f in os.listdir(directory)]
