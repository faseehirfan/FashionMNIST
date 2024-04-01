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

# Plot single image

for image, label in testing_set.take(1):
        break

image = image.numpy().reshape((28, 28))
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

# Set up layers of the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(40, (2, 2), padding='same', activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(60, (2, 2), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(80, (2, 2), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation=tf.nn.relu),

    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compiling the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

BATCH_SIZE = 32

# Shuffling and grouping data into batches 
training_set = training_set.cache().repeat().shuffle(len(training_set)).batch(BATCH_SIZE)
testing_set = testing_set.cache().batch(BATCH_SIZE)
