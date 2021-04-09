import collections
import itertools
import numpy as np
import tensorflow as tf
# import tensorflow_federated as tff
import pickle


def create_keras_model():
    network = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(28 * 28,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    network.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                    loss=tf.keras.losses.categorical_crossentropy,
                    metrics=['accuracy'])
    return network

# def create_keras_model():
#     # model building
#     model = tf.keras.models.Sequential()
#     # convolutional layer with rectified linear unit activation
#     model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
#                                      activation='relu',
#                                      input_shape=(28, 28, 1)))
#     # 32 convolution filters used each of size 3x3
#     # again
#     model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#     # model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
#     # 64 convolution filters used each of size 3x3
#     model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
#     # choose the best features via pooling
#     model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#     # randomly turn neurons on and off to improve convergence
#     # flatten since too many dimensions, we only want a classification output
#     model.add(tf.keras.layers.Flatten())
#     # fully connected to get all relevant data
#     model.add(tf.keras.layers.Dense(128, activation='relu'))
#     model.add(tf.keras.layers.Dropout(0.5))
#     # output a softmax to squash the matrix into output probabilities
#     model.add(tf.keras.layers.Dense(10, activation='softmax'))

#     model.compile(loss=tf.keras.losses.categorical_crossentropy,
#                   optimizer=tf.keras.optimizers.SGD(
#                       learning_rate=0.01),
#                   metrics=['accuracy'])
#     return model


# (trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()
# # reshape dataset to have a single channel
# trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
# testX = testX.reshape((testX.shape[0], 28, 28, 1))
# # one hot encode target values
# trainY = tf.keras.utils.to_categorical(trainY)
# testY = tf.keras.utils.to_categorical(testY)

# trainX = trainX.astype('float32')
# testX = testX.astype('float32')
# # normalize to range 0-1
# trainX = trainX / 255.0
# testX = testX / 255.0
