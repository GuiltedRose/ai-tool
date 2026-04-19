import os
import numpy as np
from scipy import signal
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from layer import Layer
from activation import Activate, Tanh, Sigmoid
from dense import Dense
from convolution import Convolution
from losses import cross_entropy, cross_entropy_prime
from reshape import Reshape
import time

class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    def backward(self, output_gradient, learning_rate):
       n = np.size(self.output)
       tmp = np.tile(self.output, n)
       return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)

def preprocess_data(x, y, limit):
    x = x[:limit]
    y = y[:limit]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x, y

WEIGHTS_FILE = "weights.npz"

# Load MSINT from server, limit to 100 images per class
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 60000)
x_test, y_test = preprocess_data(x_test, y_test, 10000)

network = [
        Convolution((1, 28, 28), 3, 5),
        Sigmoid(),
        Reshape((5, 26, 26), (5 * 26 * 26, 1)),
        Dense(5 * 26 * 26, 100),
        Sigmoid(),
        Dense(100, 10),
        Softmax()
]

# Loading Weights
if os.path.exists(WEIGHTS_FILE):
    data = np.load(WEIGHTS_FILE)
    dense_layers = [l for l in network if isinstance(l, Dense)]
    for i, layer in enumerate(dense_layers):
        layer.weights = data[f'dense_weights_{i}']
        layer.bias = data[f'dense_bias_{i}']
    conv_layers = [l for l in network if isinstance(l, Convolution)]
    for i, layer in enumerate(conv_layers):
        layer.kernels = data[f'conv_kernels_{i}']
        layer.biases = data[f'conv_biases_{i}']

epochs = 500
learning_rate = 0.0001
start_time = time.time()

def save_weights():
   # Saving Weights
    dense_layers = [l for l in network if isinstance(l, Dense)]
    conv_layers = [l for l in network if isinstance(l, Convolution)]
    np.savez(WEIGHTS_FILE,
        **{f'dense_weights_{i}': l.weights for i, l in enumerate(dense_layers)},
        **{f'dense_bias_{i}': l.bias for i, l in enumerate(dense_layers)},
        **{f'conv_kernels_{i}': l.kernels for i, l in enumerate(conv_layers)},
        **{f'conv_biases_{i}': l.biases for i, l in enumerate(conv_layers)}
    )

# Training
for e in range(epochs):
    epoch_start = time.time()
    error = 0
    for x, y in zip(x_train, y_train):
        output = x
        for layer in network:
            output = layer.forward(output)
        error += cross_entropy(y, output)
        grad = cross_entropy_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)
    error /= len(x_train)
    epoch_time = time.time() - epoch_start
    total_time = time.time() - start_time
    eta = (total_time / (e + 1)) * (epochs - (e + 1))
    save_weights()
    print(f"{e + 1}/{epochs}, error={error}, epoch_time={epoch_time:.2f}s, total_time={total_time:.2f}s, eta={eta:.2f}s")

# Testing
for x, y in zip(x_test, y_test):
    output = x
    for layer in network:
        output = layer.forward(output)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")

