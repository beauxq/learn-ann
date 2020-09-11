from abc import ABC, abstractmethod
from random import random
import numpy as np

class Layer:
    class Activation(ABC):
        @staticmethod
        @abstractmethod
        def f(x):
            pass

        @staticmethod
        @abstractmethod
        def der(x):
            pass

    class Sigmoid(Activation):
        @staticmethod
        def f(x):
            return 1 / (1 + np.exp(-x))

        @staticmethod
        def der(x):
            sx = Layer.Sigmoid.f(x)
            return sx * (1 - sx)

    class ReLU(Activation):
        @staticmethod
        def f(x):
            return np.maximum(0, x)

        @staticmethod
        def der(x):
            return 1 * (x > 0)

    def __init__(self, neuron_count: int, neuron_count_previous: int, activation: type):
        self._weights = np.array(
            [[random() for __ in range(neuron_count)] for _ in range(neuron_count_previous)]
        )
        self._biases = np.array([random() for _ in range(neuron_count)])
        self.activation: type = activation

        self._weighted_sum_before_activation: np.ndarray = np.ndarray((0,))  # z
        self._output: np.ndarray = np.ndarray((0,))  # after activation function

        self._input: np.ndarray = np.ndarray((0,))  # from previous layer
        self._dirty = True  # output needs to be recalculated

        # all the derivatives
        self.derror_dout: np.ndarray = np.ndarray((0,))  # der of error with respect to output
        self.dout_din: np.ndarray = np.ndarray((0,))  # der of output with respect to input
        self.din_dw: np.ndarray = np.ndarray((0,))  # der of input with respect to weights
        self.derror_din: np.ndarray = np.ndarray((0,))  # der of error with respect to input
        self.derror_dw: np.ndarray = np.ndarray((0,))  # der of error with respect to weights

    def __repr__(self):
        to_return = "Layer weights:\n"
        to_return += str(self._weights) + "\nbiases:\n" + str(self._biases)
        return to_return

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
        return self._weights

    @weights.setter
    def weights(self, value: np.ndarray):
        self._weights = value
        self._dirty = True

    @property
    def neuron_count(self):
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
        # print(self._derror_din)
        for one_for_each_input_set in self.derror_din:
            self._biases -= learning_rate * one_for_each_input_set
        self._dirty = True
