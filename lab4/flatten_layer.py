from numpy.typing import NDArray

from layer import Layer


class FlattenLayer(Layer):
    def __init__(self, input_shape=None):
        super().__init__('', input_shape=input_shape)

    def get_output_shape(self):
        return (
            self._input_shape[0] * self._input_shape[1] * self._input_shape[2]
        )

    def forward(self, x: NDArray):
        x = x.flatten()
        return x.reshape(x.shape[0], 1)

    def backward(self, e: NDArray):
        return e.reshape(self._input_shape)

    def update_weight(self, learning_rate: float):
        pass
