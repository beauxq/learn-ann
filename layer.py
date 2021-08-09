from abc import ABC, abstractmethod
from typing import Any, List, Union, Optional, Dict, Type
import numpy as np

from util import ndarray2str, str2ndarray

class Layer:
    class Activation(ABC):
        @staticmethod
        @abstractmethod
        def f(x: np.ndarray) -> np.ndarray:
            """ activation function """

        @staticmethod
        @abstractmethod
        def der(x: np.ndarray) -> np.ndarray:
            """ derivative of activation function """

    class Sigmoid(Activation):
        """ sigmoid activation function """
        @staticmethod
        def f(x: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-x))

        @staticmethod
        def der(x: np.ndarray) -> np.ndarray:
            sx = Layer.Sigmoid.f(x)
            return sx * (1 - sx)

    class TanH(Activation):
        """ tanh activation function """
        @staticmethod
        def f(x: np.ndarray) -> np.ndarray:
            ex = np.exp(x)
            enx = np.exp(-x)
            return (ex - enx) / (ex + enx)

        @staticmethod
        def der(x: np.ndarray) -> np.ndarray:
            return 1 - np.power(Layer.TanH.f(x), 2)

    class ReLU(Activation):
        """ rectified linear unit activation function """
        @staticmethod
        def f(x: np.ndarray) -> np.ndarray:
            return np.maximum(0, x)

        @staticmethod
        def der(x: np.ndarray) -> np.ndarray:
            return 1 * (x > 0)

    class ELU(Activation):
        """ exponential linear unit
        not parametric """
        @staticmethod
        def f(x: np.ndarray) -> np.ndarray:
            return (x >= 0) * x + (x < 0) * (np.exp(x) -1)

        @staticmethod
        def der(x: np.ndarray) -> np.ndarray:
            return np.minimum(np.exp(x), 1)

    class Swish(Activation):
        """ Swish https://arxiv.org/pdf/1710.05941v2.pdf """
        b = 1

        @staticmethod
        def f(x: np.ndarray) -> np.ndarray:
            return x * Layer.Sigmoid.f(Layer.Swish.b * x)

        @staticmethod
        def der(x: np.ndarray) -> np.ndarray:
            b_swish = Layer.Swish.b * Layer.Swish.f(x)
            return b_swish + Layer.Sigmoid.f(Layer.Swish.b * x) * (1 - b_swish)

    class TruncatedSQRT(Activation):
        """ truncated square root """
        origin_slope = 4
        assert origin_slope > 0
        b = 1 / (origin_slope * 2)
        a = b * b

        @staticmethod
        def f(x: np.ndarray) -> np.ndarray:
            # possible low level optimization:
            # use a bitmask to take only the sign bit of x
            # "xor" or "or" with the result of (sqrt(abs(x) + a) - b)
            return np.sign(x) * (np.sqrt(np.abs(x) + Layer.TruncatedSQRT.a) - Layer.TruncatedSQRT.b)

        @staticmethod
        def der(x: np.ndarray) -> np.ndarray:
            return 1 / (2 * np.sqrt(np.abs(x) + Layer.TruncatedSQRT.a))

    class SQRT(Activation):
        """ square root (not truncated) """

        @staticmethod
        def f(x: np.ndarray) -> np.ndarray:
            return np.sign(x) * (np.sqrt(np.abs(x)))

        @staticmethod
        def der(x: np.ndarray) -> np.ndarray:
            return 1 / (2 * np.sqrt(np.abs(x)) + 1e-307)

    class SqrtToLinear(Activation):
        """ truncated sqrt when negative,
        linear when positive """
        linear_slope = 1
        assert linear_slope > 0
        b = 1 / (linear_slope * 2)
        a = b * b

        @staticmethod
        def f(x: np.ndarray) -> np.ndarray:
            return ((x >= 0) * (Layer.SqrtToLinear.linear_slope * x)
                    + (x < 0) * (-(np.sqrt(np.abs(x) + Layer.SqrtToLinear.a)
                                   - Layer.SqrtToLinear.b)))

        @staticmethod
        def der(x: np.ndarray) -> np.ndarray:
            return ((x >= 0) * (Layer.SqrtToLinear.linear_slope)
                    + (x < 0) * (1 / (2 * np.sqrt(np.abs(x) + Layer.SqrtToLinear.a))))

    def __init__(self,
                 neuron_count_or_string_list: Union[int, List[str]],
                 neuron_count_previous: Optional[int] = None,
                 activation: Optional[Type["Layer.Activation"]] = None,
                 random_init: bool = True):
        # overloaded
        if isinstance(neuron_count_or_string_list, int):
            assert isinstance(neuron_count_previous, int)
            assert isinstance(activation, type)
            assert issubclass(activation, Layer.Activation)
            neuron_count = neuron_count_or_string_list
            self._weights: np.ndarray = np.random.rand(neuron_count_previous, neuron_count) * 4 - 2
            if not random_init:
                interval = 4.0 / (neuron_count_previous * neuron_count)
                v = -1.995
                for row in range(neuron_count_previous):
                    for col in range(neuron_count):
                        self._weights[row][col] = v
                        v += interval
            self._biases: np.ndarray = np.random.rand(neuron_count) * 4 - 2
            if not random_init:
                self._biases = np.zeros((1, neuron_count))
            self.activation: Type["Layer.Activation"] = activation
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
        # TODO: make a warning or something if this activation isn't in layer because I can't load it
        # or make a system that can save the functions in a string
        # (in case someone subclasses Layer.Activation without modifying Layer)
        to_return = "layer " + str(self.activation.__name__) + "\n"
        to_return += "weights:\n" + ndarray2str(self._weights) + \
            "\nbiases:\n" + ndarray2str(self._biases)
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
        self.weights = str2ndarray(weights_expression)
        i += 1  # skip biases label
        biases_expression = ""
        while i < len(repr_list):
            biases_expression += repr_list[i]
            i += 1
        self._biases = str2ndarray(biases_expression)

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
        # if np.max(self.derror_dw) < 1e-6 and np.max(self.derror_dz) < 1e-6:
        #     print("update is 0")
        self._dirty = True

    def __getstate__(self) -> Dict[str, Any]:
        """ This is called on the source of Python's `deepcopy` """
        return {
            'activation': self.activation,
            '_weights': np.copy(self._weights),
            '_biases': np.copy(self._biases)
        }

    def __setstate__(self, state: Dict[str, Any]):
        """ This is called on the destination of Python's `deepcopy` """
        self.activation = state['activation']
        self._weights = state['_weights']
        self._biases = state['_biases']
        self._dirty = True

    def mutate(self, max_amount: float):
        """ change layer randomly (for evolution algorithms) """
        _range = max_amount * 2
        self._weights += np.random.rand(self._weights.shape[0], self._weights.shape[1]) \
            * _range - max_amount
        self._biases += np.random.rand(self._biases.shape[0]) \
            * _range - max_amount
        self._dirty = True
