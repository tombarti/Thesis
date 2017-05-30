import os
import util
import imghdr
from tqdm import tqdm

# so that imghdr recognises jpeg with ICC_PROFILE header
def test_icc_profile_images(h, f):
  if h.startswith(b'\xff\xd8') and h[6:17] == b'ICC_PROFILE':
    return "jpeg"

imghdr.tests.append(test_icc_profile_images)

DIR = './images/'
fnames = util.get_fnames_in_dir(DIR)

n_removed = 0
n_renamed = 0

for f in tqdm(fnames):
  t_img = imghdr.what(file=DIR + f, h=None)
  if not t_img:
    os.remove(DIR + f)
    n_removed += 1
    continue
  if t_img != 'jpeg':
    new_f = os.path.splitext(f)[0] + '.' + t_img
    os.rename(DIR+f, DIR+new_f)
    n_renamed += 1

print("Removed: {}".format(n_removed))
print("Renamed: {}".format(n_renamed))
  
