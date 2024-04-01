# FashionMNIST

A basic CNN built with TensorFlow which classifies articles of clothing from the Fashion MNIST dataset with a 91% test accuracy.

The following architecture was implemented in the model:

Input (28, 28, 1) → Convolution (2, 2, 40) → Max Pooling (2, 2) → Convolution (2, 2, 60) → Max Pooling (2, 2) → Convolution (2, 2, 80) → Max Pooling (2, 2) → Flatten → Dense (128) → Output (10)
