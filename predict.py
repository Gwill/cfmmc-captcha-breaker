# coding: utf8

import PIL
import numpy as np
import network
from configs import *


def predict(model, img):
    X = np.zeros((1, HEIGHT, WIDTH, NUM_CHANNELS), dtype=np.uint8)
    X[0] = img
    y = model.predict(X)
    vericode = network.decode(y)
    return vericode

