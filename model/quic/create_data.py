import collections
import itertools
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import pickle

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()


def preprocess(dataset):

    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        return (tf.reshape(element['pixels'], [-1, 784]), tf.reshape(element['label'], [-1, 1]))

    def generate_array():
        for element in dataset.map(batch_format_fn):
            yield element[0][0], element[1][0][0]
    x, y = zip(*generate_array())
    return np.array(x), np.array(y)


data = dict()
i = 0
for client_id in emnist_train.client_ids:
    print(client_id, '       ', i)
    x_train, y_train = preprocess(emnist_train.create_tf_dataset_for_client(
        client_id))
    y_train = tf.keras.utils.to_categorical(y_train)
    data[client_id] = (x_train, y_train)
    i += 1

with open('EMNINST.pickle', 'wb') as fp:
    print('writing data')
    pickle.dump(data, fp)
