import tensorflow_federated as tff
import collections
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, InputLayer, Softmax
from tensorflow.keras.models import Sequential

DATASET_PATH = "./WESAD/"
NUM_CLIENTS = 15
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 200
PREFETCH_BUFFER = 10
NUM_ROUNDS = 20


def read_data(DATASET_PATH):

    train_dataset_dict = collections.OrderedDict()
    test_dataset_dict = collections.OrderedDict()

    for root, dirnames, filenames in os.walk(DATASET_PATH, topdown=True):
        for file_name in filenames :
            
            if file_name.endswith('.npy'):
                subject_no = file_name.split('_')[0]
                file_path = os.path.join(root,file_name)

                if os.path.exists(file_path):           
                    data = np.load(file_path, allow_pickle=True)
                    print(f'Number of examples in subject {subject_no}: ',data.item().get('features').shape[0])
                    if file_name.endswith('train_data.npy'):
                        train_dataset_dict[subject_no] = collections.OrderedDict((('features', data.item().get('features')),('labels',data.item().get('labels'))))
                    elif file_name.endswith('test_data.npy'):
                        test_dataset_dict[subject_no] = collections.OrderedDict((('features', data.item().get('features')),('labels',data.item().get('labels'))))

    return train_dataset_dict, test_dataset_dict

def convert_to_client_dataset(train_data, test_data):
    tff_train_dataset = tff.simulation.datasets.TestClientData(train_data)
    tff_test_dataset = tff.simulation.datasets.TestClientData(test_data)

    return tff_train_dataset, tff_test_dataset

# update this method afterwards
def plot_each_subject():

    # Number of examples per layer for a sample of clients
    f = plt.figure(figsize=(12, 7))
    f.suptitle('Label Counts for a Sample of Clients')
    for i in range(6):
    client_dataset = tff_dataset.create_tf_dataset_for_client(
        tff_dataset.client_ids[i])
    plot_data = collections.defaultdict(list)
    for example in client_dataset:
        # Append counts individually per label to make plots
        # more colorful instead of one color per plot.
        print(example)
        label = example['labels'].numpy()
        plot_data[label].append(label)
    plt.subplot(2, 3, i+1)
    plt.title('Client {}'.format(i))
    for j in range(10):
        plt.hist(
            plot_data[j],
            density=False,
            bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

def preprocess_fn(dataset):

  def batch_format_fn(element,epsilon=1e-8):
    """Flatten a batch `features` and return the features as an `OrderedDict`."""

    mean, variance = tf.nn.moments(element['features'], axes=[0])
    element['features']  = (element['features']  - mean) / tf.sqrt(variance + epsilon) 
    
    return collections.OrderedDict(
        x=tf.reshape(element['features'], [-1, 32]),
        y=tf.reshape(element['labels'], [-1, 1]))

  return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1).batch(
      BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


def make_federated_data(client_data, client_ids):
  return [
      preprocess_fn(client_data.create_tf_dataset_for_client(x))
      for x in client_ids
  ]

def create_keras_model():

  return tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(32,)),
      # tf.keras.layers.Dense(15, kernel_initializer='glorot_uniform'),
      # tf.keras.layers.Dense(20, kernel_initializer='zeros'),
      tf.keras.layers.Dense(10,kernel_initializer='glorot_uniform', ),
      # kernel_regularizer=tf.keras.regularizers.l2(.0001)),
      tf.keras.layers.Softmax()
      
  ])


def model_fn():
  # We _must_ create a new model here, and _not_ capture it from an external
  # scope. TFF will call this within different graph contexts.
  keras_model = create_keras_model()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=data.element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

def build_fed_avg_process(federated_train_data):

    iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.001),
    server_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.01))
    state = iterative_process.initialize()
    
    for round_num in range(2, NUM_ROUNDS):
        state, metrics = iterative_process.next(state, federated_train_data)
        print('round {:2d}, metrics={}'.format(round_num, metrics))

    
def run_fed_evaluation(tff_test_data):
    evaluation = tff.learning.build_federated_evaluation(model_fn)
    sample_clients = tff_test_data.client_ids[:NUM_CLIENTS]
    federated_test_data = make_federated_data(tff_test_data, sample_clients)

    model_weights = iterative_process.get_model_weights(state)
    test_metrics = evaluation(model_weights, federated_test_data)
    test_metrics