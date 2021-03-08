import collections
import itertools
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import pickle
from matplotlib import pyplot as plt
from fl_model import create_keras_model

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()


def preprocess(dataset):
    '''
    Preprocess for CNN
    '''
    def batch_format_fn(element):
        return (tf.reshape(element['pixels'], [28, 28, 1]), element['label'])

    def generate_array():
        for element in dataset.map(batch_format_fn):
            yield element[0], element[1]
    x, y = zip(*generate_array())
    return np.array(x), np.array(y)


model = create_keras_model()
for client_id in emnist_train.client_ids:
    print(client_id)
    x_train, y_train = preprocess(emnist_train.create_tf_dataset_for_client(
        client_id))
    y_train = tf.keras.utils.to_categorical(y_train)
    model.fit(x_train, y_train, epochs=5, batch_size=40, validation_split=0.2)

for client_id in emnist_test.client_ids:
    print(client_id)
    x_train, y_train = preprocess(emnist_train.create_tf_dataset_for_client(
        client_id))
    y_train = tf.keras.utils.to_categorical(y_train)
    score = model.evaluate(x_train, y_train, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

# data = dict()
# i = 0
# for client_id in emnist_train.client_ids:
#     print(client_id, '       ', i)
#     x_train, y_train = preprocess(emnist_train.create_tf_dataset_for_client(
#         client_id))
#     y_train = tf.keras.utils.to_categorical(y_train)
#     data[client_id] = (x_train, y_train)
#     i += 1

# with open('EMNINST_TRAIN.pickle', 'wb') as fp:
#     print('writing train  data')
#     pickle.dump(data, fp)

# data = dict()
# i = 0
# for client_id in emnist_test.client_ids:
#     print(client_id, '       ', i)
#     x_train, y_train = preprocess(emnist_train.create_tf_dataset_for_client(
#         client_id))
#     y_train = tf.keras.utils.to_categorical(y_train)
#     data[client_id] = (x_train, y_train)
#     i += 1

# with open('EMNINST_TEST.pickle', 'wb') as fp:
#     print('writing test data')
#     pickle.dump(data, fp)
