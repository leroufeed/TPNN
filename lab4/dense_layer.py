from numpy.typing import NDArray

from layer import Layer
import numpy as np


class DenseLayer(Layer):
    def __init__(self, neurons_count: int, activation: str = 'none', input_shape=None):
        super().__init__(activation, input_shape=input_shape)
        self._neurons_count = neurons_count
        self._weight_matrix = np.empty(0)
        self._shift = np.empty(0)
        self._sum = np.empty(0)
        self._x = np.empty(0)

        self._d_weight_matrix_buffer = np.empty(0)
        self._d_shift_buffer = np.empty(0)

    def get_output_shape(self):
        return self._neurons_count

    def forward(self, x):
        self._x = x
        self._sum = np.matmul(self._weight_matrix, x) + self._shift
        return self._activation_function(self._sum)

    def backward(self, e):
        d_sum = e * self._d_activation_function(self._sum)
        d_matrix = np.matmul(d_sum, self._x.T)
        self._d_weight_matrix_buffer += d_matrix
        self._d_shift_buffer += d_sum
        return np.matmul(self._weight_matrix.T, d_sum)

    def update_weight(self, learning_rate: float):
        self._weight_matrix -= learning_rate * self._d_weight_matrix_buffer
        self._shift -= learning_rate * self._d_shift_buffer

        self._d_weight_matrix_buffer = np.zeros(self._weight_matrix.shape)
        self._d_shift_buffer = np.zeros(self._shift.shape)

    def init(self):
        self._weight_matrix = np.random.random((self.get_output_shape(), self._input_shape)) * 2 - 1
        self._shift = np.random.random((self.get_output_shape(), 1)) * 2 - 1

        self._d_weight_matrix_buffer = np.zeros(self._weight_matrix.shape)
        self._d_shift_buffer = np.zeros(self._shift.shape)
