import os, sys
from keras.models import load_model, Sequential
from keras.layers import Activation, Layer
import numpy as np
import imageio
from keras.datasets import cifar10
from tensorflow.keras import optimizers
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ ==  '__main__':
    
    modelname = 'cifer_vgg16_lens_flare_acc86_77_ir0_98_asr95_6'
    modelpath = './models/'+modelname+'.h5'
    model = load_model(modelpath)
    model.summary()
    layers = model.layers
    model2 = Sequential()
    for layer in layers:
        # print( layer.get_config()['activation'] )
        if 'activation' in layer.get_config().keys():
            act_name = layer.get_config()['activation']
            config = layer.get_config()
            config.pop('activation')
            model2.add(layer.__class__.from_config(config))
            model2.add(Activation(act_name))
        else:
            model2.add(layer)
    model2.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(), metrics=['accuracy'])
    model2.summary()
    model2.load_weights(modelpath)
    model2.save('./models/'+modelname+'_2.h5')
    model_yaml = model2.to_json()
    with open(modelname+".json", "w") as yaml_file:
        yaml_file.write(model_yaml)

