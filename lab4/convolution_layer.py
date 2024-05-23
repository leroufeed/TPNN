from copy import deepcopy

import numpy as np

from layer import Layer


class ConvolutionLayer(Layer):
    def __init__(self, filters_count: int, kernel_size: tuple[int, int], activation: str, input_shape=None):
        super().__init__(activation, input_shape=input_shape)
        self._filters_count = filters_count
        self._kernel_size = kernel_size
        self._x = np.empty(0)
        self._sum = np.empty(0)

        self._kernels = np.empty(0)
        self._shift = np.empty(0)

        self._d_kernels_buffer = np.empty(0)
        self._d_shift_buffer = np.empty(0)

    def get_output_shape(self):
        return (self._filters_count,
                self._input_shape[1] - self._kernel_size[0] + 1,
                self._input_shape[2] - self._kernel_size[1] + 1)

    def forward(self, x):
        self._x = x
        out = ConvolutionLayer.convolution(x, self._kernels)
        out += self._shift
        self._sum = deepcopy(out)
        return self._activation_function(out)

    @staticmethod
    def convolution(tensor, kernels):
        output_size = (
            kernels.shape[0],
            tensor.shape[1] - kernels.shape[2] + 1,
            tensor.shape[2] - kernels.shape[3] + 1
        )
        out = np.zeros(output_size)
        for k in range(output_size[0]):
            for i in range(output_size[1]):
                for j in range(output_size[2]):
                    out[k][i][j] = np.sum(
                        tensor[:, i:i + kernels.shape[2], j:j + kernels.shape[3]] * kernels[k], axis=(0, 1, 2)
                    )
        return out

    def backward(self, e):
        d_sum = e * self._d_activation_function(self._sum)
        rotated_kernel = np.flip(np.flip(self._kernels, 2), 1)
        padding_size = (self._kernel_size[1] - 1, self._kernel_size[0] - 1)
        d_sum_padding = np.pad(d_sum,
                               ((0, 0), padding_size, padding_size), 'constant', constant_values=0
                               )

        rotated_kernel = np.transpose(rotated_kernel, (1, 0, 2, 3))

        for i in range(self._kernels.shape[0]):
            d_kernels_i = ConvolutionLayer.convolution(self._x, np.array([[d_sum[i]]]))
            self._d_kernels_buffer[i] += d_kernels_i
        self._d_shift_buffer += d_sum

        return ConvolutionLayer.convolution(d_sum_padding, rotated_kernel)

    def update_weight(self, learning_rate: float):
        learning_rate *= 0.01
        self._kernels -= learning_rate * self._d_kernels_buffer
        self._shift -= learning_rate * self._d_shift_buffer

        self._d_kernels_buffer = np.zeros(self._kernels.shape)
        self._d_shift_buffer = np.zeros(self._shift.shape)

    def init(self):
        self._kernels = (np.random.random((
            self._filters_count, self._input_shape[0],
            self._kernel_size[0], self._kernel_size[1])
        ) - 0.5) * 0.5
        self._shift = (np.random.random((self.get_output_shape())) - 0.5) * 0.5

        self._d_kernels_buffer = np.zeros(self._kernels.shape)
        self._d_shift_buffer = np.zeros(self._shift.shape)

