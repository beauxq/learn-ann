from typing import List, Optional
import numpy as np
from layer import Layer

class Network:
    """
    artificial neural network

    can be copied with Python's `deepcopy` for evolution algorithms
    """
    def __init__(self, input_feature_count: int = 2):
        self._input_feature_count = input_feature_count
        self._layers: List[Layer] = []

    def add_layer(self, neuron_count: int, activation: type, random_init: bool=True):
        if not issubclass(activation, Layer.Activation):
            raise ValueError("activation should be subclass of Layer.Activation")
        neuron_count_previous = len(self._layers[-1].weights[0]) \
            if len(self._layers) > 0 \
            else self._input_feature_count
        self._layers.append(Layer(neuron_count, neuron_count_previous, activation, random_init))

    def _gradient_descent(self,
                          input_sets: np.ndarray,
                          target_output: np.ndarray,
                          learning_rate: float):
        # backpropagation
        for i, layer in reversed(tuple(enumerate(self._layers))):
            layer.derror_dout = layer.output - target_output \
                if i == len(self._layers) - 1 \
                else np.dot(self._layers[i+1].derror_dz, self._layers[i+1].weights.T)
            # print("on last layer:", i == len(self._layers) - 1)
            # print("layer.output", layer.output, sep="/n")
            # print("target_output", target_output, sep="/n")
            # if i != len(self._layers) - 1:
            #     print("i+1 de_dz:", self._layers[i+1].derror_dz, sep="\n")
            # print("derror_dout:", layer.derror_dout, sep="/n")
            # activation type is asserted in Layer ctor
            layer.dout_dz = layer.activation.der(layer.weighted_sum_before_activation)  # type: ignore
            # print("dout_dz:", layer.dout_dz, sep="\n")
            layer.dz_dw = input_sets if i == 0 else self._layers[i-1].output
            # print("dz_dw:", layer.dz_dw, "\n")

            layer.derror_dz = layer.derror_dout * layer.dout_dz
            layer.derror_dw = np.dot(layer.dz_dw.T, layer.derror_dz)

        # update weights
        for layer in self._layers:
            layer.update(learning_rate)
    
    def _mse(self, target_output: np.ndarray) -> float:
        """ mean squared error """
        squared_error: np.ndarray = np.power(target_output - self._layers[-1].output, 2)
        # print("target:\n", target_output)
        # print("output:\n", self._layers[-1].output)
        # print("differ:\n", target_output - self._layers[-1].output)
        # print("square:\n", squared_error)

        # np.mean gives a float whether target is 1d or 2d - type checker doesn't like it
        return np.mean(squared_error)  # type: ignore
    
    def verify_shapes(self, input_sets: np.ndarray, target_output: Optional[np.ndarray]=None) -> None:
        """ raises exception if dimensions don't match """
        if input_sets.shape[1] != self._input_feature_count:
            raise ValueError(
                "input doesn't have the right feature count - " +
                str(input_sets.shape) + " " + str(self._layers[0].weights.shape)
            )
        if target_output is not None:
            if input_sets.shape[0] != target_output.shape[0]:
                raise ValueError("input and output sizes don't match - shapes " +
                                str(input_sets.shape) + " " + str(target_output.shape))
            if target_output.shape[1] != self._layers[-1].neuron_count:
                raise ValueError("target_output shape doesn't match output layer - " +
                                str(target_output.shape[1]))

    def train(self,
              input_sets: np.ndarray,
              target_output: np.ndarray,
              epoch_count: int,
              learning_rate: float,
              report_every: int = 2000) -> float:
        """
        `report_every` can be `0` to never report error values,
        which means `report_every` can be `bool` for
        never report error values
        or
        always report error values
        
        returns the ending mean squared error
        """
        self.verify_shapes(input_sets, target_output)

        for epoch in range(epoch_count):
            for i, layer in enumerate(self._layers):
                # print("layer i", i)
                # print("layer", layer)
                layer.input = self._layers[i - 1].output if i > 0 else input_sets
                # print(layer.input)

            # report error this epoch
            if (report_every > 0) and (epoch % report_every) == 0:
                error_mean = self._mse(target_output)
                print("error mean:", error_mean)
            self._gradient_descent(input_sets, target_output, learning_rate)

        return self._mse(target_output)

    def predict(self, input_set: np.ndarray) -> np.ndarray:
        """ return prediction based on current model """
        if len(input_set.shape) < 2:
            input_set = np.array([input_set])
        self.verify_shapes(input_set)

        self._layers[0].input = input_set
        for i in range(1, len(self._layers)):
            self._layers[i].input = self._layers[i-1].output
        return self._layers[-1].output

    def mutate(self, rate: float):
        """ change network randomly """
        for layer in self._layers:
            layer.mutate(rate)

    def __repr__(self):
        to_return = "features " + str(self._input_feature_count)
        for layer in self._layers:
            to_return += "\n" + str(layer)
        return to_return

    def load_from_str(self, repr_: str):
        """ invert the __repr__ function
        to produce the network from the string produced by __repr__ """
        lines = repr_.splitlines()
        self._input_feature_count = int(lines[0][9:])
        self._layers = []
        i = 1
        while i < len(lines):
            group_for_layer = [lines[i]]
            i += 1
            while (i < len(lines)) and not lines[i].startswith("layer"):
                group_for_layer.append(lines[i])
                i += 1
            self._layers.append(Layer(group_for_layer))

# I can't see any reason to use softmax as an activation function
# I can't even find any claims that it helps with accuracy, much less any evidence.
# Lots of people talk about how useful it is,
#   but that's not a reason to put it in a neural network.
# If you want softmax, take your output from the neural network,
#   and put it through a softmax function.
def softmax(x: np.ndarray) -> np.ndarray:
    """ stable softmax """
    shift_x = x - np.max(x)
    ex = np.exp(shift_x)
    return ex / np.sum(ex)

def _test():
    print(softmax(np.array([0.2, 0.5, 0.3])))
    # It doesn't even do the job well...
    # If I already have a probability distribution, why change it?

if __name__ == "__main__":
    _test()
