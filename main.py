from typing import List
import numpy as np
from layer import Layer
from network import Network

def main():
    # 4 input sets, 2 input features
    input_sets = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # an output for each input set to train on
    target_output = np.array([[0], [1], [1], [0]])  # xor

    neuron_counts_in_hidden_layers = [5, 3]
    hidden_activation = Layer.Sigmoid
    epoch_count = 40000
    learning_rate = 0.0625

    test_input = np.array([[0, 1]])


    net = Network(len(input_sets[0]))
    for neuron_count in neuron_counts_in_hidden_layers:
        net.add_layer(neuron_count, hidden_activation)
    net.add_layer(len(target_output[0]), Layer.Sigmoid)

    net.train(input_sets, target_output, epoch_count, learning_rate)

    print(net.predict(test_input))


if __name__ == "__main__":
    main()
