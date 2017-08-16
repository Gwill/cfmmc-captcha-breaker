# coding: utf8

import network
import h5py
import numpy as np


h5file = h5py.File('captchas/dataset.h5', 'r')
X = np.asarray(h5file['X'])
y = np.asarray(h5file['y'])

model = network.create_model()

model.fit(X, y, batch_size=32, epochs=1, validation_split=0.1)

model.save_weights('my_model_weights_real_data.h5')

