# coding: utf8

import PIL
from configs import *


def predict(img):
    X = np.zeros((1, HEIGHT, WIDTH, NUM_CHANNELS), dtype=np.uint8)
    X[0] = img
    y = model.predict(X)
    vericode = network.decode(y)
    return vericode

