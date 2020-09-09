from random import random
import numpy as np

EPOCH_COUNT = 50000

input_features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target_output = np.array([[0], [1], [1], [0]])  # xor

neuron_counts_in_hidden_layers = [5, 3]

test_input = np.array([[0, 1]])

class Layer:
    def __init__(self, neuron_count: int, neuron_count_previous: int):
        self.weights = np.array(
            [[random() for __ in range(neuron_count)] for _ in range(neuron_count_previous)]
        )
        self.dot_weights_before_activation = None  # applied weights but not activation function yet
        self.output = None  # after activation function

        # all the derivatives
        self.derror_dout = None  # derivative of error with respect to output
        self.dout_din = None  # derivative of output with respect to input
        self.din_dw = None  # derivative of input with respect to weights
        self.derror_din = None  # derivative of error with respect to input
        self.derror_dw = None  # derivative of error with respect to weights


def main():
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_der(x):
        sx = sigmoid(x)
        return sx * (1 - sx)

    hidden_layer_count = len(neuron_counts_in_hidden_layers)
    layer_count = hidden_layer_count + 1
    layers = []
    for i in range(layer_count):
        neuron_count = neuron_counts_in_hidden_layers[i] \
            if i < hidden_layer_count \
            else len(target_output[0])
        neuron_count_previous = len(layers[-1].weights[0]) \
            if i > 0 \
            else len(input_features[0])
        layers.append(Layer(neuron_count, neuron_count_previous))

    # bias = random()
    learning_rate = 0.0625

    for epoch in range(EPOCH_COUNT):
        for i, layer in enumerate(layers):
            # print("layer", layer)
            output_from_previous = layers[i - 1].output if i > 0 else input_features
            # print(output_from_previous)
            layer.dot_weights_before_activation = \
                np.dot(output_from_previous, layer.weights)
            layer.output = sigmoid(layer.dot_weights_before_activation)

        # report error this epoch
        if epoch % 40 == 0:
            mean_sq_error_o = ((target_output - layers[-1].output) ** 2) / 2
            error_sum = mean_sq_error_o.sum()
            print("error sum:", error_sum)

        # back propogation
        for i, layer in reversed(tuple(enumerate(layers))):
            layer.derror_dout = layer.output - target_output \
                if i == layer_count - 1 \
                else np.dot(layers[i+1].derror_din, layers[i+1].weights.T)
            layer.dout_din = sigmoid_der(layer.dot_weights_before_activation)
            layer.din_dw = input_features if i == 0 else layers[i-1].output

            layer.derror_din = layer.derror_dout * layer.dout_din
            layer.derror_dw = np.dot(layer.din_dw.T, layer.derror_din)

        # update weights
        for layer in layers:
            # print("update weights")
            # print(layer.weights)
            # print(layer.derror_dw)
            layer.weights -= learning_rate * layer.derror_dw

        """
        # update bias
        for i in derror_dino:
            bias -= learning_rate * i
        """

    out_from_layer = []
    for i, layer in enumerate(layers):
        previous_layer_output = test_input if i == 0 else out_from_layer[-1]
        out_from_layer.append(sigmoid(np.dot(previous_layer_output, layer.weights)))
        print(layer.weights)

    for each in out_from_layer:
        print(each)

if __name__ == "__main__":
    main()
