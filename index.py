import math
import numpy as np
import matplotlib.pyplot as plt
import logging

import tensorflow as tf
import tensorflow_datasets as tfds


# Error logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Function to normalize pixel values to be between [0, 1]
def normalize_data(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label

# tfds.disable_progress_bar()

# Loading Fashion MNIST dataset
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
training_set, testing_set = dataset['train'].map(normalize_data), dataset['test'].map(normalize_data)