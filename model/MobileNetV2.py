# Keras implementation of MobileNet v2

# Python Libraries
import os
import warnings
# 3rd Party Libraries
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.keras_applications.mobilenetv2 import MobileNetV2
from tensorflow.keras.model import Model

def build_model(size, alpha):
    input_tensor = Input(shape=(size, size, 3))
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(size, size, 3),
        pooling='avg')

    for layer in base_model.layers:
        layer.trainable = False  # trainable has to be false in order to freeze the layers

    op = Dense(256, activation='relu')(base_model.output)
    op = Dropout(.25)(op)

    ##
    # softmax: calculates a probability for every possible class.
    #
    # activation='softmax': return the highest probability;
    # for example, if 'Coat' is the highest probability then the result would be
    # something like [0,0,0,0,1,0,0,0,0,0] with 1 in index 5 indicate 'Coat' in our case.
    ##
    output_tensor = Dense(10, activation='softmax')(op)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model
