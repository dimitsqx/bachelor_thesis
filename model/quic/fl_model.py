import collections
import itertools
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import pickle


def create_keras_model():
    network = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return network
