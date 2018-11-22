# Keras implementation of MobileNet v2

# Python Libraries
import os
import warnings
# 3rd Party Libraries
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.keras_applications.mobilenetv2 import MobileNetV2
from tensorflow.keras.model import Model


def build_model(size: tuple, alpha: float) -> Model:
    """
    Builds a Keras Model for MobileNetV2
    :param size: a tuple containing the dimensions of the image (X, (Y, (C)))
    :param alpha: a float that specifies the width multiplier, greater alpha produces a larger network
    :return: returns a Keras Model of MobileNetV2
    :rtype: Model
    """
    # size can be either a scalar or tuple
    # assume size refers to a square color image input
    if len(size) == 1:
        try:
            X = Y = size[0]
        except (TypeError, IndexError):
            X = Y = size
        C = 3
    # Assume size is an X by Y color image input
    elif len(size) == 2:
        X, Y = size
        C = 3
    # Size is an X by Y of C channel image input
    else:
        X, Y, C = size

    # Input shape given autoconfiguration
    input_tensor = Input(shape=(X, Y, C))
    # Use Keras applications MobileNetV2 with pretrained ImageNet weights
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(X, Y, C),
        pooling='avg')

    # MobileNetV2 is frozen with ImageNet weights
    for layer in base_model.layers:
        layer.trainable = False

    # Add a retrainable FCN with RelU activation
    op = Dense(256, activation='relu')(base_model.output)
    # ... and dropout
    op = Dropout(.25)(op)

    # Then a softmax classification layer, change first parameter to the number of classes
    output_tensor = Dense(10, activation='softmax')(op)
    # model will include all layers required in the computation of
    # output_tensor given input_tensor.
    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model
