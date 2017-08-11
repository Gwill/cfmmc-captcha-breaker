# coding: utf8

import network_ctc

base_model, model = network_ctc.create_model()

evaluator = network_ctc.Evaluator()

model.fit_generator(network_ctc.gen(), samples_per_epoch=51200, nb_epoch=100, 
                    nb_worker=52, pickle_safe=True, callbacks=[evaluator])

model.save('my_model_ctc.h5')
model.save_weights('my_model_ctc_weights.h5')

