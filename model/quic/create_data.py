import collections
import itertools
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import pickle
from matplotlib import pyplot as plt
from fl_model import create_keras_model

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()


# def preprocess(dataset):
#     '''
#     Preprocess for CNN
#     '''
#     def batch_format_fn(element):
#         return (tf.reshape(element['pixels'], [28, 28, 1]), element['label'])

#     def generate_array():
#         for element in dataset.map(batch_format_fn):
#             yield element[0], element[1]
#     x, y = zip(*generate_array())
#     return np.array(x), np.array(y)

def preprocess(dataset):

    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        return (tf.reshape(element['pixels'], [-1, 784]), tf.reshape(element['label'], [-1, 1]))

    def generate_array():
        for element in dataset.map(batch_format_fn):
            yield element[0][0], element[1][0][0]
    x, y = zip(*generate_array())
    return np.array(x), np.array(y)


# data = emnist_train.create_tf_dataset_for_client('f0052_42')
# print(len(data))
# a, b = preprocess(data)
# print(len(a), len(b))
# b = tf.keras.utils.to_categorical(b, num_classes=10)
model = create_keras_model()
# model.fit(a, b, epochs=5, batch_size=40, validation_split=0.2)

for client_id in emnist_train.client_ids:
    print(client_id)
    x_train, y_train = preprocess(emnist_train.create_tf_dataset_for_client(
        client_id))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    model.fit(x_train, y_train, epochs=5, batch_size=40, validation_split=0.2)
    break

for client_id in emnist_test.client_ids:
    print(client_id)
    x_train, y_train = preprocess(emnist_test.create_tf_dataset_for_client(
        client_id))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    score = model.evaluate(x_train, y_train, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    break

# data = dict()
# i = 0
# for client_id in emnist_train.client_ids:
#     print(client_id, '       ', i)
#     x_train, y_train = preprocess(emnist_train.create_tf_dataset_for_client(
#         client_id))
#     y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
#     data[client_id] = (x_train, y_train)
#     i += 1

# with open('model\\quic\\EMNINST_TRAIN_2NN.pickle', 'wb') as fp:
#     print('writing train  data')
#     pickle.dump(data, fp)

# data = dict()
# i = 0
# for client_id in emnist_test.client_ids:
#     print(client_id, '       ', i)
#     x_train, y_train = preprocess(emnist_train.create_tf_dataset_for_client(
#         client_id))
#     y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
#     data[client_id] = (x_train, y_train)
#     i += 1

# with open('model\\quic\\EMNINST_TEST_2NN.pickle', 'wb') as fp:
#     print('writing test data')
#     pickle.dump(data, fp)
