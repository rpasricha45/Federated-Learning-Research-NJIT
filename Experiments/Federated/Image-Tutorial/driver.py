# NOTE: Jupyter requires a patch to asyncio.

import collections
import warnings
from six.moves import range
from tensorflow_federated.python.simulation import HDF5ClientData
import numpy as np
import six
import tensorflow as tf
import tensorflow_federated as tff
from lower import MnistTrainableModel
from model_examples import TrainableLinearRegression



emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
myclient = HDF5ClientData("student31.h5")

example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])
userDataset =myclient.create_tf_dataset_for_client(myclient.client_ids[0])

NUM_CLIENTS = 10
NUM_EPOCHS = 10
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500

def preprocess(dataset):
  def element_fn(element):
    return collections.OrderedDict([
        ('x', tf.reshape(element['x'], [-1])),
        ('y', tf.reshape(element['y'], [1])),
    ])
  return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
      SHUFFLE_BUFFER).batch(BATCH_SIZE)
preprocessed_example_dataset = preprocess(userDataset)

sample_batch = tf.nest.map_structure(
    lambda x: x.numpy(), iter(preprocessed_example_dataset).next())

def make_federated_data(client_data, client_ids):
  return [preprocess(client_data.create_tf_dataset_for_client(x))
          for x in client_ids]

sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

federatedData = make_federated_data(myclient,myclient.client_ids)

# federated_train_data = make_federated_data(emnist_train, sample_clients)
iterative_process = tff.learning.build_federated_averaging_process(TrainableLinearRegression)

state = iterative_process.initialize()

state, metrics = iterative_process.next(state, federatedData)
print('round  1, metrics={}'.format(metrics))

for round_num in range(2, 8):
  state, metrics = iterative_process.next(state, federatedData)
  print('round {:2d}, metrics={}'.format(round_num, metrics))
