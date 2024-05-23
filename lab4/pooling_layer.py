import numpy as np

from layer import Layer


class PoolingLayer(Layer):
    def __init__(self, size, input_shape=None):
        self._size = size
        self._maximum_indexes = np.empty(0)
        self._x_shape = ()
        super().__init__('', input_shape=input_shape)

    def get_output_shape(self):
        return (self._input_shape[0],
                int(self._input_shape[1] / self._size),
                int(self._input_shape[2] / self._size))

    def forward(self, x):
        out_shape = self.get_output_shape()
        out = np.zeros(out_shape)
        self._maximum_indexes = np.zeros((out_shape[0], out_shape[1], out_shape[2], 2))
        self._x_shape = x.shape
        for k in range(out_shape[0]):
            for i in range(out_shape[1]):
                for j in range(out_shape[2]):
                    sub_matrix = x[k][self._size * i:self._size * i + self._size,
                                 self._size * j:self._size * j + self._size]
                    maximum = np.max(sub_matrix)
                    out[k][i][j] = maximum
                    index = sub_matrix.argmax()
                    x1, y1 = int(index / self._size), index % self._size
                    self._maximum_indexes[k][i][j] = [self._size * i + x1,
                                                      self._size * j + y1]


        return out

    def backward(self, e):
        out = np.zeros(self._x_shape)
        out_shape = self.get_output_shape()
        for k in range(out_shape[0]):
            for i in range(out_shape[1]):
                for j in range(out_shape[2]):
                    indexes = self._maximum_indexes[k][i][j]
                    out[k][int(indexes[0])][int(indexes[1])] = e[k][i][j]

        return out

    def update_weight(self, learning_rate: float):
        pass
