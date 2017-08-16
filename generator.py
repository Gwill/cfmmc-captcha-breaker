# coding: utf8
import numpy as np
import CaptchaGenerator.generate_captcha as capgen
from configs import *


def gen(batch_size=64):
    X = np.zeros((batch_size, HEIGHT, WIDTH, NUM_CHANNELS), dtype=np.uint8)
    y = [np.zeros((batch_size, NUM_OF_CLASSES), dtype=np.uint8) for i in range(NUM_OF_LABELS)]
    while True:
        for i in range(batch_size):
            image, label = capgen.create_validate_code()
            X[i] = image
            for j, ch in enumerate(label):
                y[j][i, :] = 0
                y[j][i, CHARS[ch]] = 1
        yield X, y