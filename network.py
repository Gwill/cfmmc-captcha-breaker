# coding: utf8
import numpy as np
import keras as K
import CaptchaGenerator.generate_captcha as capgen
from configs import *


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

