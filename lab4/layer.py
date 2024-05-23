from numpy.typing import NDArray

from activation_functions import name_to_functions


class Layer:
    def __init__(self, activation: str, input_shape=None):
        self._input_shape: tuple = input_shape
        self._activation_function, self._d_activation_function = name_to_functions[activation]

    def get_input_shape(self):
        return self._input_shape

    def get_output_shape(self):
        pass

    def set_input_size(self, input_shape: tuple):
        self._input_shape = input_shape

    def forward(self, x):
        pass

    def backward(self, e):
        pass

    def update_weight(self, learning_rate: float):
        pass

    def init(self):
        pass
