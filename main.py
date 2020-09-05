from random import random
import numpy as np

EPOCH_COUNT = 50000

input_features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target_output = np.array([[0, 1, 1, 0]]).T  # xor

neuron_counts_in_hidden_layers = [5, 3]

hidden_layer_count = len(neuron_counts_in_hidden_layers)
weights = []
for layer in range(len(neuron_counts_in_hidden_layers) + 1):
    neuron_count = neuron_counts_in_hidden_layers[layer] \
        if layer < hidden_layer_count \
        else len(target_output[0])
    neuron_count_previous = len(weights[-1][0]) \
        if layer > 0 \
        else len(input_features[0])
    weights.append(np.array(
        [[random() for __ in range(neuron_count)] for _ in range(neuron_count_previous)]
    ))
for weight in weights:
    print(weight)

# bias = random()
learning_rate = 0.0625

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

layer_count = hidden_layer_count + 1
for epoch in range(EPOCH_COUNT):
    dot_weights_before_activation = []
    output_from = []
    for layer in range(layer_count):
        # print("layer", layer)
        output_from_previous = output_from[-1] if layer > 0 else input_features
        # print(output_from_previous)
        dot_weights_before_activation.append(
            np.dot(output_from_previous, weights[layer])
        )
        output_from.append(sigmoid(dot_weights_before_activation[layer]))
    """
    dot_weights_before_activation_0 = np.dot(input_features, weights_0)
    output_from_0 = sigmoid(dot_weights_before_activation_0)

    dot_weights_before_activation_1 = np.dot(output_from_0, weights_1)
    output_from_1 = sigmoid(dot_weights_before_activation_1)
    """

    # report error this epoch
    if epoch % 40 == 0:
        mean_sq_error_o = ((target_output - output_from[-1]) ** 2) / 2
        error_sum = mean_sq_error_o.sum()   
        print("error sum:", error_sum)

    # back propogation
    derror_dout = [1 for _ in range(layer_count)]  # derivative of error with respect to output
    dout_din = [1 for _ in range(layer_count)]  # derivative of output with respect to input
    din_dw = [1 for _ in range(layer_count)]  # derivative of input with respect to weights
    derror_din = [1 for _ in range(layer_count)]  # derivative of error with respect to input
    derror_dw = [1 for _ in range(layer_count)]  # derivative of error with respect to weights

    for i in range(layer_count - 1, -1, -1):
        derror_dout[i] = output_from[i] - target_output \
            if i == layer_count - 1 \
            else np.dot(derror_din[i+1], weights[i+1].T)
        dout_din[i] = sigmoid_der(dot_weights_before_activation[i])
        din_dw[i] = input_features if i == 0 else output_from[i-1]

        derror_din[i] = derror_dout[i] * dout_din[i]
        derror_dw[i] = np.dot(din_dw[i].T, derror_din[i])

    """
    # phase 1 backprop
    derror_dout_1 = output_from_1 - target_output
    dout_din_1 = sigmoid_der(dot_weights_before_activation_1)
    din_dw_1 = output_from_0

    derror_din_1 = derror_dout_1 * dout_din_1
    derror_dw_1 = np.dot(din_dw_1.T, derror_din_1)

    # phase 0 backprop
    # derror_dwh = derror_dout_0 * dout_din_0 * din_dw_0
    # dino_douth = weights_1
    derror_dout_0 = np.dot(derror_din_1, weights_1.T)
    dout_din_0 = sigmoid_der(dot_weights_before_activation_0)
    din_dw_0 = input_features

    derror_dw_0 = np.dot(din_dw_0.T, dout_din_0 * derror_dout_0)
    """

    # update weights
    for i in range(layer_count):
        weights[i] -= learning_rate * derror_dw[i]
    """
    weights_0 -= learning_rate * derror_dw_0
    weights_1 -= learning_rate * derror_dw_1
    """

    """
    # update bias
    for i in derror_dino:
        bias -= learning_rate * i
    """

# print(weights)
# print(bias)

test_input = np.array([[0, 1]])
out_from_layer = []
for layer in range(layer_count):
    previous_layer_output = test_input if layer == 0 else out_from_layer[-1]
    out_from_layer.append(sigmoid(np.dot(previous_layer_output, weights[layer])))
    print(weights[layer])

for each in out_from_layer:
    print(each)
