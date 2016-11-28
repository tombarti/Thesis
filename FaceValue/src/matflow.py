import scipy.io as sio

DIR  = 'matlab/'

net_desc = sio.loadmat(DIR+'alexnet-face-bn.mat')

for k, v in net_desc.items():
  print(k)

net_layers = net_desc['layers'][0]
for l in net_layers:
  print(l['type'])
  print(l.dtype.names)
