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
        self.weights = np.array(
            [[random() for __ in range(neuron_count)] for _ in range(neuron_count_previous)]
        )
        self.biases = np.array([random() for _ in range(neuron_count)])
        self.activation = activation

        self.weighted_sum_before_activation = None  # z
        self.output = None  # after activation function

        # all the derivatives
        self.derror_dout = None  # derivative of error with respect to output
        self.dout_din = None  # derivative of output with respect to input
        self.din_dw = None  # derivative of input with respect to weights
        self.derror_din = None  # derivative of error with respect to input
        self.derror_dw = None  # derivative of error with respect to weights

    def __repr__(self):
        to_return = "Layer weights:\n"
        to_return += str(self.weights) + "\nbiases:\n" + str(self.biases)
        return to_return
