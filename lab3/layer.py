from numpy.typing import NDArray

from RNN.activation_functions import name_to_functions


class Layer:
    def __init__(self, activation: str, input_shape=None):
        self._input_size: int = input_shape
        self._activation_function, self._d_activation_function = name_to_functions[activation]

    def get_input_shape(self):
        return self._input_size

    def get_output_shape(self):
        pass

    def set_input_size(self, input_size: int):
        self._input_size = input_size

    def forward(self, x):
        pass

    def backward(self, e, learning_rate: float):
        pass

    def init(self):
        pass
