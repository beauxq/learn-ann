from abc import ABC, abstractmethod
from random import random
from typing import List, Union, Optional
import numpy as np

class Layer:
    class Activation(ABC):
        @staticmethod
        @abstractmethod
        def f(x):
            """ activation function """

        @staticmethod
        @abstractmethod
        def der(x):
            """ derivative of activation function """

    class Sigmoid(Activation):
        """ sigmoid activation function """
        @staticmethod
        def f(x):
            return 1 / (1 + np.exp(-x))

        @staticmethod
        def der(x):
            sx = Layer.Sigmoid.f(x)
            return sx * (1 - sx)

    class ReLU(Activation):
        """ rectified linear unit activation function """
        @staticmethod
        def f(x):
            return np.maximum(0, x)

        @staticmethod
        def der(x):
            return 1 * (x > 0)

    def __init__(self,
                 neuron_count_or_string_list: Union[int, List[str]],
                 neuron_count_previous: Optional[int] = None,
                 activation: Optional[type] = None):
        # overloaded
        if isinstance(neuron_count_or_string_list, int):
            neuron_count = neuron_count_or_string_list
            self._weights = np.array(
                [
                    [random() - 0.5 for __ in range(neuron_count)]
                    for _ in range(neuron_count_previous)
                ]
            )
            self._biases = np.array([random() - 0.5 for _ in range(neuron_count)])
            self.activation: type = activation
        else:  # string list
            self.load_from_strings(neuron_count_or_string_list)

        # commonly labeled "z" in information on ANNs
        self._weighted_sum_before_activation: np.ndarray = np.ndarray((0,))
        # commonly labeled "a" in information on ANNs
        # after activation function
        self._output: np.ndarray = np.ndarray((0,))

        # output from previous layer
        self._input: np.ndarray = np.ndarray((0,))
        self._dirty = True  # output needs to be recalculated

        # all the derivatives
        self.derror_dout: np.ndarray = np.ndarray((0,))  # der of error with respect to output
        self.dout_dz: np.ndarray = np.ndarray((0,))  # der of output with respect to z
        self.dz_dw: np.ndarray = np.ndarray((0,))  # der of z with respect to weights
        self.derror_dz: np.ndarray = np.ndarray((0,))  # der of error with respect to z
        self.derror_dw: np.ndarray = np.ndarray((0,))  # der of error with respect to weights

    def __repr__(self):
        to_return = "layer " + str(self.activation.__name__) + "\n"
        to_return += "weights:\n" + repr(self._weights) + "\nbiases:\n" + repr(self._biases)
        return to_return

    def load_from_strings(self, repr_list: List[str]):
        """ invert the __repr__ function
        to produce the network from a list of string lines produced by __repr__ """
        activ = repr_list[0][6:]
        self.activation = getattr(Layer, activ)
        assert repr_list[1] == "weights:"
        i = 2
        weights_expression = ""
        while (i < len(repr_list)) and (repr_list[i] != "biases:"):
            weights_expression += repr_list[i]
            i += 1
        # TODO: SECURITY ISSUE: parse these instead of using eval
        self.weights = eval("np." + weights_expression)
        i += 1  # skip biases label
        biases_expression = ""
        while i < len(repr_list):
            biases_expression += repr_list[i]
            i += 1
        self._biases = eval("np." + biases_expression)

    @property
    def output(self):
        """ reading this value will calculate the output
        if it's not already calculated """
        if self._dirty:
            self._calculate_output()
        return self._output

    @property
    def weighted_sum_before_activation(self):
        """ reading this value will calculate the output
        if it's not already calculated """
        if self._dirty:
            self._calculate_output()
        return self._weighted_sum_before_activation

    @property
    def input(self):
        """ output from previous layer """
        return self._input

    @input.setter
    def input(self, value: np.ndarray):
        self._input = value
        self._dirty = True

    @property
    def weights(self):
        """ the matrix of weights between the previous layer and this layer """
        return self._weights

    @weights.setter
    def weights(self, value: np.ndarray):
        self._weights = value
        self._dirty = True

    @property
    def neuron_count(self):
        """ the number of neurons in this layer """
        return len(self._biases)

    def _calculate_output(self):
        """ calculate weighted sum before activation
        and output after activation """
        self._weighted_sum_before_activation = \
            np.dot(self._input, self._weights) + self._biases
        # print("before biases:")
        # print(np.dot(self._input, self._weights))
        # print("after biases:")
        # print(self._weighted_sum_before_activation)
        self._output = self.activation.f(self._weighted_sum_before_activation)
        self._dirty = False

    def update(self, learning_rate: float):
        """ weights and biases
            precondition: set derivatives """
        # print("update weights")
        # print(self._weights)
        # print(self._derror_dw)
        self._weights -= learning_rate * self.derror_dw
        # print("update biases")
        # print(self._biases)
        # print(self._derror_dz)
        for one_for_each_input_set in self.derror_dz:
            self._biases -= learning_rate * one_for_each_input_set
        self._dirty = True
