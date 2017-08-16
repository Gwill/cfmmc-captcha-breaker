# coding: utf8

import h5py
import numpy as np
import PIL
import glob
from configs import *


files = glob.glob('captchas/*.jpg')
dataset_size = len(files)
X = np.zeros((dataset_size, HEIGHT, WIDTH, NUM_CHANNELS), dtype=np.uint8)
y = [np.zeros((dataset_size, NUM_OF_CLASSES), dtype=np.uint8) for i in range(NUM_OF_LABELS)]

for i in range(dataset_size):
    file = files[i]
    img = PIL.Image.open(file)
    label = file.replace('captchas\\', '').replace('.jpg', '')
    X[i] = img
    for j, ch in enumerate(label):
        y[j][i, :] = 0
        y[j][i, CHARS[ch]] = 1

h5file = h5py.File('captchas/dataset.h5', 'w')
h5file['X'] = X
h5file['y'] = y
h5file.close()

aaa = h5py.File('captchas/dataset.h5', 'r')
xx = aaa['X']
yy = aaa['y']