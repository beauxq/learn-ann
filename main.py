from copy import deepcopy
import numpy as np
from layer import Layer
from network import Network

GD = False  # gradient descent or evolution

def main():
    # 4 input sets, 2 input features
    input_sets = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # an output for each input set to train on
    target_output = np.array([[0], [1], [1], [0]])  # xor
    input_feature_count = len(input_sets[0])
    output_feature_count = len(target_output[0])

    neuron_counts_in_hidden_layers = [5, 3]
    hidden_activation = Layer.TruncatedSQRT
    epoch_count = 40000
    learning_rate = 0.0625 if GD else 8.0

    test_input = np.array([[1, 1], [0, 1], [0, 0], [1, 0]])

    net = Network(input_feature_count)
    for neuron_count in neuron_counts_in_hidden_layers:
        net.add_layer(neuron_count, hidden_activation)
    net.add_layer(output_feature_count, Layer.Sigmoid)

    if GD:
        net.train(input_sets, target_output, epoch_count, learning_rate)
    else:
        # evolution
        for epoch in range(epoch_count):
            netcopy = deepcopy(net)
            netcopy.mutate(learning_rate)
            net_error = np.abs(target_output - net.predict(input_sets)).sum()
            cop_error = np.abs(target_output - netcopy.predict(input_sets)).sum()
            if epoch % 2000 == 0:
                print(net_error, cop_error)
            if cop_error < net_error:
                net = netcopy
                if cop_error < 0.00001:
                    print("cutting training at epoch", epoch)
                    break

    # test save and load in string
    # print(net)
    saved = str(net)
    new_net = Network()
    new_net.load_from_str(saved)

    print(new_net)
    print("results:")
    print(new_net.predict(test_input))


if __name__ == "__main__":
    main()
