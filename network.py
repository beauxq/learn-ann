from typing import List
import numpy as np
from layer import Layer

class Network:
    def __init__(self, input_feature_count: int):
        self._input_feature_count = input_feature_count
        self._layers: List[Layer] = []

    def add_layer(self, neuron_count: int, activation: type):
        neuron_count_previous = len(self._layers[-1].weights[0]) \
            if len(self._layers) > 0 \
            else self._input_feature_count
        self._layers.append(Layer(neuron_count, neuron_count_previous, activation))

    def train(self,
              input_sets: np.ndarray,
              target_output: np.ndarray,
              epoch_count: int,
              learning_rate: float):
        # TODO: test all of these exceptions
        if input_sets.shape[1] != self._input_feature_count:
            raise ValueError(
                "input_sets doesn't have the right feature count - " +
                "input_sets.shape: " + str(input_sets.shape) +
                " - network input feature count: " + str(self._layers[0].weights.shape)
            )
        if input_sets.shape[0] != target_output.shape[0]:
            raise ValueError("input and output sizes don't match - shapes " +
                             str(input_sets.shape) + " " + str(target_output.shape))
        if target_output.shape[1] != self._layers[-1].neuron_count:
            raise ValueError("target_output shape doesn't match output layer - " +
                             str(target_output.shape[1]))
        for epoch in range(epoch_count):
            for i, layer in enumerate(self._layers):
                # print("layer", layer)
                layer.input = self._layers[i - 1].output if i > 0 else input_sets
                # print(layer.input)

            # report error this epoch
            if epoch % 40 == 0:
                mean_sq_error_o = ((target_output - self._layers[-1].output) ** 2) / 2
                error_sum = mean_sq_error_o.sum()
                print("error sum:", error_sum)

            # backpropagation
            for i, layer in reversed(tuple(enumerate(self._layers))):
                layer.derror_dout = layer.output - target_output \
                    if i == len(self._layers) - 1 \
                    else np.dot(self._layers[i+1].derror_din, self._layers[i+1].weights.T)
                layer.dout_din = layer.activation.der(layer.weighted_sum_before_activation)
                layer.din_dw = input_sets if i == 0 else self._layers[i-1].output

                layer.derror_din = layer.derror_dout * layer.dout_din
                layer.derror_dw = np.dot(layer.din_dw.T, layer.derror_din)

            # update weights
            for layer in self._layers:
                layer.update(learning_rate)

    def predict(self, input_set: np.ndarray) -> np.ndarray:
        """ return prediction based on current model """
        self._layers[0].input = input_set
        for i in range(1, len(self._layers)):
            self._layers[i].input = self._layers[i-1].output
        return self._layers[-1].output
