import numpy as np

from layer import Layer

EPOCH_COUNT = 50000

# 4 input sets, 2 input features
input_sets = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# an output for each input set to train on
target_output = np.array([[0], [1], [1], [0]])  # xor

neuron_counts_in_hidden_layers = [5, 3]
hidden_activation = Layer.ReLU

test_input = np.array([[0, 1]])


def main():
    hidden_layer_count = len(neuron_counts_in_hidden_layers)
    layer_count = hidden_layer_count + 1
    layers = []
    for i in range(layer_count):
        neuron_count = neuron_counts_in_hidden_layers[i] \
            if i < hidden_layer_count \
            else len(target_output[0])
        neuron_count_previous = len(layers[-1].weights[0]) \
            if i > 0 \
            else len(input_sets[0])  # number of input features
        activation = hidden_activation \
            if i < hidden_layer_count \
            else Layer.Sigmoid
        layers.append(Layer(neuron_count, neuron_count_previous, activation))

    # bias = random()
    learning_rate = 0.0625

    for epoch in range(EPOCH_COUNT):
        for i, layer in enumerate(layers):
            # print("layer", layer)
            output_from_previous = layers[i - 1].output if i > 0 else input_sets
            # print(output_from_previous)
            layer.weighted_sum_before_activation = \
                np.dot(output_from_previous, layer.weights) + layer.biases
            # print("before biases:")
            # print(np.dot(output_from_previous, layer.weights))
            # print("after biases:")
            # print(layer.weighted_sum_before_activation)
            layer.output = layer.activation.f(layer.weighted_sum_before_activation)

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
            layer.dout_din = layer.activation.der(layer.weighted_sum_before_activation)
            layer.din_dw = input_sets if i == 0 else layers[i-1].output

            layer.derror_din = layer.derror_dout * layer.dout_din
            layer.derror_dw = np.dot(layer.din_dw.T, layer.derror_din)

        # update weights
        for layer in layers:
            # print("update weights")
            # print(layer.weights)
            # print(layer.derror_dw)
            layer.weights -= learning_rate * layer.derror_dw
            # print("update biases")
            # print(layer.biases)
            # print(layer.derror_din)
            for one_for_each_input_set in layer.derror_din:
                layer.biases -= learning_rate * one_for_each_input_set

    out_from_layer = []
    for i, layer in enumerate(layers):
        previous_layer_output = test_input if i == 0 else out_from_layer[-1]
        out_from_layer.append(layer.activation.f(
            np.dot(previous_layer_output, layer.weights) + layer.biases
        ))
        print(layer.weights)

    for each in out_from_layer:
        print(each)

if __name__ == "__main__":
    main()
