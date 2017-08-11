# coding: utf8
import numpy as np
import keras as K
import CaptchaGenerator.generate_captcha as capgen
from configs import *


rnn_size = 128


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)


def gen(batch_size=32):
    X = np.zeros((batch_size, WIDTH, HEIGHT, NUM_CHANNELS), dtype=np.uint8)
    y = np.zeros((batch_size, NUM_OF_LABELS), dtype=np.uint8)
    while True:
        for i in range(batch_size):
            image, label = capgen.create_validate_code()
            X[i] = np.array(image).transpose(1, 0, 2)
            y[i] = [CHARS[x] for x in label]
        yield [X, y, np.ones(batch_size) * 10, 
               np.ones(batch_size)*NUM_OF_LABELS], np.ones(batch_size)


def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([INIT_CHARS[x] for x in y])


def create_model(training=True):
    input_tensor = K.layers.Input((WIDTH, HEIGHT, 3))
    x = input_tensor
    for i in range(2):
        x = K.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = K.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        x = K.layers.MaxPooling2D((2, 2))(x)

    conv_shape = x.get_shape()
    x = K.layers.Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)
    x = K.layers.Dense(32, activation='relu')(x)

    gru_1 = K.layers.GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')(x)
    gru_1b = K.layers.GRU(rnn_size, return_sequences=True, go_backwards=True, 
                              init='he_normal', name='gru1_b')(x)
    gru1_merged = K.layers.merge([gru_1, gru_1b], mode='sum')

    gru_2 = K.layers.GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
    gru_2b = K.layers.GRU(rnn_size, return_sequences=True, go_backwards=True, 
                              init='he_normal', name='gru2_b')(gru1_merged)
    x = K.layers.merge([gru_2, gru_2b], mode='concat')

    if training:
        x = K.layers.Dropout(0.25)(x)
    else:
        x = K.layers.Dropout(0)(x)
    
    x = K.layers.Dense(NUM_OF_CLASSES, init='he_normal', activation='softmax')(x)
    base_model = K.models.Model(input=input_tensor, output=x)

    labels = K.layers.Input(name='the_labels', shape=[NUM_OF_LABELS], dtype='float32')
    input_length = K.layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = K.layers.Input(name='label_length', shape=[1], dtype='int64')
    loss_out = K.layers.Lambda(ctc_lambda_func, output_shape=(1,), 
                                        name='ctc')([x, labels, input_length, label_length])

    model = K.models.Model(input=[input_tensor, labels, input_length, label_length], output=[loss_out])
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
    return base_model, model

def evaluate(model, batch_num=10):
    batch_acc = 0
    generator = gen()
    for i in range(batch_num):
        [X_test, y_test, _, _], _  = next(generator)
        y_pred = base_model.predict(X_test)
        shape = y_pred[:,2:,:].shape
        ctc_decode = K.ctc_decode(y_pred[:,2:,:], 
                                  input_length=np.ones(shape[0])*shape[1])[0][0]
        out = K.get_value(ctc_decode)[:, :6]
        if out.shape[1] == 6:
            batch_acc += ((y_test == out).sum(axis=1) == 6).mean()
    return batch_acc / batch_num

class Evaluator(K.callbacks.Callback):
    def __init__(self):
        self.accs = []
    
    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(base_model)*100
        self.accs.append(acc)
        print('')
        print('acc: %f%%'%batch_acc)

