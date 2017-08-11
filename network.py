# coding: utf8
import numpy as np
import keras as K
import CaptchaGenerator.generate_captcha as capgen
from configs import *


def gen(batch_size=32):
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


def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([INIT_CHARS[x] for x in y])


def create_model(training=True):
    input_tensor = K.Input((HEIGHT, WIDTH, 3))
    x = input_tensor
    for i in range(4):
        x = K.layers.Conv2D(32*2**i, 3, activation='relu', padding='same')(x)
        x = K.layers.Conv2D(32*2**i, 3, activation='relu', padding='same')(x)
        x = K.layers.MaxPooling2D((2, 2))(x)

    x = K.layers.Flatten()(x)
    if training:
        x = K.layers.Dropout(0.25)(x)
    else:
        x = K.layers.Dropout(0)(x)
    x = [K.layers.Dense(NUM_OF_CLASSES, activation='softmax', name='c%d'%(i+1))(x) for i in range(NUM_OF_LABELS)]
    model = K.models.Model(inputs=input_tensor, outputs=x)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    return model

