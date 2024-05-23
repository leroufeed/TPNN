import numpy as np
from numpy.typing import NDArray

from layer import Layer


def softmax(x: NDArray[float]) -> NDArray[float]:
    x = x - np.max(x)
    exp_array = np.exp(x)
    return exp_array / np.sum(exp_array)


class ConvolutionNeuralNetwork:
    def __init__(self):
        self._layers: list[Layer] = []

    def add_layer(self, layer: Layer):
        if len(self._layers) != 0:
            layer.set_input_size(self._layers[-1].get_output_shape())
        self._layers.append(layer)

    def fit(self, x, y):
        z = self.predict(x)
        e = z - y
        error = np.sum(np.abs(e))
        for i in range(len(self._layers)):
            e = self._layers[-1 - i].backward(e)
        return error

    def update_weight(self, learning_rate: float):
        for i in range(len(self._layers)):
            self._layers[-1 - i].update_weight(learning_rate)

    def predict(self, x):
        for layer in self._layers:
            x = layer.forward(x)
        return softmax(x)

    def init(self):
        for layer in self._layers:
            layer.init()
