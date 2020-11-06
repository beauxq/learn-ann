from abc import ABC, abstractmethod
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

    class TanH(Activation):
        """ tanh activation function """
        @staticmethod
        def f(x):
            ex = np.exp(x)
            enx = np.exp(-x)
            return (ex - enx) / (ex + enx)

        @staticmethod
        def der(x):
            return 1 - np.power(Layer.TanH.f(x), 2)

    class ReLU(Activation):
        """ rectified linear unit activation function """
        @staticmethod
        def f(x):
            return np.maximum(0, x)

        @staticmethod
        def der(x):
            return 1 * (x > 0)

    class ELU(Activation):
        """ exponential linear unit
        not parametric """
        @staticmethod
        def f(x):
            return (x >= 0) * x + (x < 0) * (np.exp(x) -1)

        @staticmethod
        def der(x):
            return np.minimum(np.exp(x), 1)

    class Swish(Activation):
        """ Swish https://arxiv.org/pdf/1710.05941v2.pdf """
        b = 1

        @staticmethod
        def f(x):
            return x * Layer.Sigmoid.f(Layer.Swish.b * x)

        @staticmethod
        def der(x):
            return Layer.Swish.b * Layer.Swish.f(x) + \
                Layer.Sigmoid.f(Layer.Swish.b * x) * (1 - Layer.Swish.b * Layer.Swish.f(x))

    class TruncatedSQRT(Activation):
        """ truncated square root """
        origin_slope = 4
        assert origin_slope > 0
        b = 1 / (origin_slope * 2)
        a = b * b

        @staticmethod
        def f(x):
            # possible low level optimization:
            # use a bitmask to take only the sign bit of x
            # "xor" or "or" with the result of (sqrt(abs(x) + a) - b)
            return np.sign(x) * (np.sqrt(np.abs(x) + Layer.TruncatedSQRT.a) - Layer.TruncatedSQRT.b)

        @staticmethod
        def der(x):
            return 1 / (2 * np.sqrt(np.abs(x) + Layer.TruncatedSQRT.a))

    class SqrtToLinear(Activation):
        """ truncated sqrt when negative,
        linear when positive """
        linear_slope = 1
        assert linear_slope > 0
        b = 1 / (linear_slope * 2)
        a = b * b

        @staticmethod
        def f(x):
            return ((x >= 0) * (Layer.SqrtToLinear.linear_slope * x)
                    + (x < 0) * (-(np.sqrt(-x + Layer.SqrtToLinear.a)
                                   - Layer.SqrtToLinear.b)))

        @staticmethod
        def der(x):
            return ((x >= 0) * (Layer.SqrtToLinear.linear_slope)
                    + (x < 0) * (1 / (2 * np.sqrt(-x + Layer.SqrtToLinear.a))))

    def __init__(self,
                 neuron_count_or_string_list: Union[int, List[str]],
                 neuron_count_previous: Optional[int] = None,
                 activation: Optional[type] = None,
                 random_init = True):
        # overloaded
        if isinstance(neuron_count_or_string_list, int):
            assert neuron_count_previous is not None
            assert activation is not None
            neuron_count = neuron_count_or_string_list
            self._weights = np.random.rand(neuron_count_previous, neuron_count) * 4 - 2
            if not random_init:
                interval = 4.0 / (neuron_count_previous * neuron_count)
                v = -1.995
                for row in range(neuron_count_previous):
                    for col in range(neuron_count):
                        self._weights[row][col] = v
                        v += interval
            self._biases = np.random.rand(neuron_count) * 4 - 2
            if not random_init:
                self._biases = np.zeros((1, neuron_count))
            self.activation: type = activation
        else:  # string list
            self.load_from_strings(neuron_count_or_string_list)

        # null numpy array
        na = np.ndarray((0,))

        # commonly labeled "z" in information on ANNs
        self._weighted_sum_before_activation: np.ndarray = na
        # commonly labeled "a" in information on ANNs
        # after activation function
        self._output: np.ndarray = na

        # output from previous layer
        self._input: np.ndarray = na
        self._dirty = True  # output needs to be recalculated

        # all the derivatives
        self.derror_dout: np.ndarray = na  # of error with respect to output
        self.dout_dz: np.ndarray = na  # of output with respect to z
        self.dz_dw: np.ndarray = na  # of z with respect to weights
        self.derror_dz: np.ndarray = na  # of error with respect to z (also error with respect to bias)
        self.derror_dw: np.ndarray = na  # of error with respect to weights

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
        self._output = self.activation.f(self._weighted_sum_before_activation)  # type: ignore
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
