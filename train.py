# coding: utf8

import network

model = network.create_model()

model.fit_generator(network.gen(), samples_per_epoch=51200, nb_epoch=5, 
                    nb_worker=52, pickle_safe=True, 
                    validation_data=network.gen(), nb_val_samples=1280)

model.save('my_model.h5')
model.save_weights('my_model_weights.h5')

