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


def none(x):
    return x


def d_none(x):
    return np.ones(x.shape)


name_to_functions = {
    'relu': (relu, d_relu),
    'none': (none, d_none),
    '': (None, None)
}
