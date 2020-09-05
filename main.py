from random import random
import numpy as np

EPOCH_COUNT = 50000

input_features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target_output = np.array([[0, 1, 1, 0]]).T  # xor

neuron_counts_in_hidden_layers = [5, 3]

class Layer:
    def __init__(self, neuron_count: int, neuron_count_previous: int):
        self.weights = np.array(
            [[random() for __ in range(neuron_count)] for _ in range(neuron_count_previous)]
        )
        self.dot_weights_before_activation = None
        self.output = None

        # all the derivatives
        self.derror_dout = None  # derivative of error with respect to output
        self.dout_din = None  # derivative of output with respect to input
        self.din_dw = None  # derivative of input with respect to weights
        self.derror_din = None  # derivative of error with respect to input
        self.derror_dw = None  # derivative of error with respect to weights


hidden_layer_count = len(neuron_counts_in_hidden_layers)
layer_count = hidden_layer_count + 1
layers = []
for layer in range(layer_count):
    neuron_count = neuron_counts_in_hidden_layers[layer] \
        if layer < hidden_layer_count \
        else len(target_output[0])
    neuron_count_previous = len(layers[-1].weights[0]) \
        if layer > 0 \
        else len(input_features[0])
    layers.append(Layer(neuron_count, neuron_count_previous))

# bias = random()
learning_rate = 0.0625

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

for epoch in range(EPOCH_COUNT):
    for layer_i in range(layer_count):
        # print("layer", layer)
        output_from_previous = layers[layer_i - 1].output if layer_i > 0 else input_features
        # print(output_from_previous)
        layers[layer_i].dot_weights_before_activation = \
            np.dot(output_from_previous, layers[layer_i].weights)
        layers[layer_i].output = sigmoid(layers[layer_i].dot_weights_before_activation)

    # report error this epoch
    if epoch % 40 == 0:
        mean_sq_error_o = ((target_output - layers[-1].output) ** 2) / 2
        error_sum = mean_sq_error_o.sum()
        print("error sum:", error_sum)

    # back propogation
    for i in range(layer_count - 1, -1, -1):
        layers[i].derror_dout = layers[i].output - target_output \
            if i == layer_count - 1 \
            else np.dot(layers[i+1].derror_din, layers[i+1].weights.T)
        layers[i].dout_din = sigmoid_der(layers[i].dot_weights_before_activation)
        layers[i].din_dw = input_features if i == 0 else layers[i-1].output

        layers[i].derror_din = layers[i].derror_dout * layers[i].dout_din
        layers[i].derror_dw = np.dot(layers[i].din_dw.T, layers[i].derror_din)

    # update weights
    for i in range(layer_count):
        # print("update weights", i)
        # print(layers[i].weights)
        # print(layers[i].derror_dw)
        layers[i].weights -= learning_rate * layers[i].derror_dw

    """
    # update bias
    for i in derror_dino:
        bias -= learning_rate * i
    """

test_input = np.array([[0, 1]])
out_from_layer = []
for layer in range(layer_count):
    previous_layer_output = test_input if layer == 0 else out_from_layer[-1]
    out_from_layer.append(sigmoid(np.dot(previous_layer_output, layers[layer].weights)))
    print(layers[layer].weights)

for each in out_from_layer:
    print(each)
