from numpy.typing import NDArray

from RNN.layer import Layer
import numpy as np


class DenseLayer(Layer):
    def __init__(self, neurons_count: int, activation: str = 'none', input_size=None):
        super().__init__(activation, input_shape=input_size)
        self._neurons_count = neurons_count
        self._weight_matrix = np.empty(0)
        self._shift = np.empty(0)
        self._sum = np.empty(0)
        self._x = np.empty(0)

    def get_output_shape(self):
        return self._neurons_count

    def forward(self, x):
        self._x = x
        self._sum = np.matmul(self._weight_matrix, x) + self._shift
        return self._activation_function(self._sum)

    def backward(self, e, learning_rate: float):
        d_sum = e * self._d_activation_function(self._sum)
        d_matrix = np.matmul(d_sum, self._x.T)

        self._weight_matrix -= learning_rate * d_matrix
        self._shift -= learning_rate * d_sum

        return np.matmul(self._weight_matrix.T, d_sum)

    def init(self):
        self._weight_matrix = np.random.random((self.get_output_shape(), self._input_size)) * 2 - 1
        self._shift = np.random.random((self.get_output_shape(), 1)) * 2 - 1
