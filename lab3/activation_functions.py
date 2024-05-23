from copy import deepcopy

import numpy as np
from numpy.typing import NDArray


def relu(x: NDArray[float]) -> NDArray[float]:
    x = deepcopy(x)
    x[x < 0] = 0
    return x


def d_relu(x: NDArray[float]) -> NDArray[float]:
    x = deepcopy(x)
    x[x < 0] = 0
    x[x >= 0] = 1
    return x


def tanh(x: NDArray[float]) -> NDArray[float]:
    return np.tanh(x)


def d_tanh(x: NDArray[float]) -> NDArray[float]:
    return 1 - tanh(x) ** 2


def sigmoid(x: NDArray[float]) -> NDArray[float]:
    return np.tanh(x * 0.5) * 0.5 + 0.5


def d_sigmoid(x: NDArray[float]) -> NDArray[float]:
    sig = sigmoid(x)
    return x * (1 - x)

def none(x):
    return x


def d_none(x):
    return np.ones(x.shape)


name_to_functions = {
    'relu': (relu, d_relu),
    'tanh': (tanh, d_tanh),
    'none': (none, d_none),
    '': (None, None)
}
