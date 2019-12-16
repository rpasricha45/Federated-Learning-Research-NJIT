import collections
import warnings
from six.moves import range
import numpy as np
import six
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.simulation import client_data
from tensorflow_federated.python.simulation import HDF5ClientData
from model_examples import TrainableLinearRegression
from model_examples import LinearRegression
from core import MnistTrainableModel
from model_examples import TrainableLinearRegression
from tensorflow_federated.python.learning import model

warnings.simplefilter('ignore')

tf.compat.v1.enable_v2_behavior()

np.random.seed(0)

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
myclient = HDF5ClientData("student31.h5")
userDataset =myclient.create_tf_dataset_for_client(myclient.client_ids[0])
example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])
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
preprocessed_example_dataset = preprocess(example_dataset)

sample_batch = tf.nest.map_structure(lambda x: x.numpy(), iter(preprocessed_example_dataset).next())


def make_federated_data(client_data, client_ids):
  return [preprocess(client_data.create_tf_dataset_for_client(x))
          for x in client_ids]

sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

# federated_train_data = make_federated_data(emnist_train, sample_clients)
federatedData = make_federated_data(userDataset, userDataset.client_ids)
def create_compiled_keras_model():
	# create model
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(1, input_dim=1, kernel_initializer='normal', activation='relu'))
	model.add(tf.keras.layers.Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam',metrics=[tf.keras.metrics.MeanSquaredError()])
	return model

def model_fn():
  keras_model = create_compiled_keras_model()
  print(type(keras_model))
  return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

# iterative_process = tff.learning.build_federated_averaging_process(model_fn)
iterative_process = tff.learning.build_federated_averaging_process(TrainableLinearRegression)
state = iterative_process.initialize()
state, metrics = iterative_process.next(state,federatedData )
for round_num in range(1,2):
  state, metrics = iterative_process.next(state, federatedData)
  print('round {:2d}, metrics={}'.format(round_num, metrics))
