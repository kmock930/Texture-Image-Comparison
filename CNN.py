import tensorflow as tf;
from tensorflow.keras.layers import Dense;
import keras;
import constants;
import numpy as np;
from sklearn.preprocessing import LabelEncoder;
import math;
import joblib;
import matplotlib.pyplot as plt;
from datetime import datetime; # for timestamp

class CNN:
    model: keras.Model;

    def __init__(self):
        self.model = self.build_classifier();

    # https://towardsdatascience.com/10-minutes-to-building-a-cnn-binary-image-classifier-in-tensorflow-4e216b2034aa
    def build_classifier(self):
        tf.keras.layers.Conv2D(constants.filter, constants.cnn_kernel_size, activation=constants.activation1_cnn, input_shape=constants.input_shape_cnn);
        tf.keras.layers.MaxPooling2D(constants.pool_size, constants.strides);


        self.model = tf.keras.models.Sequential();

        for _ in range(3):
            self.model.add(tf.keras.layers.Conv2D(constants.filter, constants.cnn_kernel_size, activation=constants.activation1_cnn, input_shape=constants.input_shape_cnn));
            self.model.add(tf.keras.layers.MaxPooling2D(constants.pool_size, constants.strides));
        
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('dandelions') and 1 for the other ('grass')
        self.model.add(tf.keras.layers.Dense(round(math.log(2, constants.num_classes)), activation=constants.activation2_cnn));
        return self.model;
