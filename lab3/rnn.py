from numpy.typing import NDArray

from RNN.layer import Layer
import numpy as np


class RNN(Layer):
    def __init__(self, output_size: int, activation: str = 'none', input_size=None, return_sequence=False):
        super().__init__(activation, input_shape=input_size)
        self._hidden_size = 0
        self._output_size = output_size
        self._return_sequence = return_sequence
        self._sequence_length = 0

        self._x = np.empty(0)
        self._input_weights = np.empty(0)
        self._hidden_weights = np.empty(0)
        self._output_weights = np.empty(0)
        self._hidden_bias = np.empty(0)
        self._output_bias = np.empty(0)
        self._hidden_state = np.empty(0)

    def get_output_shape(self):
        return self._output_size

    def forward(self, x):
        self._x = x
        self._sequence_length = x.shape[0]

        hidden = np.zeros((self._hidden_size, 1))
        self._hidden_state = np.zeros((self._sequence_length, self._hidden_size, 1))

        for i in range(self._sequence_length):
            input_x = np.matmul(self._input_weights, x[i]).reshape(self._input_weights.shape[0], 1)
            hidden = input_x + np.matmul(self._hidden_weights, hidden) + self._hidden_bias
            hidden = self._activation_function(hidden)
            self._hidden_state[i] = hidden

        if self._return_sequence:
            output = np.matmul(self._output_weights, self._hidden_state) + self._output_bias
        else:
            final_hidden_state = self._hidden_state[-1]
            output = np.matmul(self._output_weights, final_hidden_state) + self._output_bias

        return output

    def backward(self, e, learning_rate: float):
        d_input_weights = np.zeros_like(self._input_weights)
        d_hidden_weights = np.zeros_like(self._hidden_weights)
        d_output_weights = np.zeros_like(self._output_weights)
        d_output_bias = np.zeros_like(self._output_bias)
        d_hidden_bias = np.zeros_like(self._hidden_bias)
        d_x = np.zeros_like(self._x)
        if len(d_x.shape) == 2:
            d_x = d_x[:, :, np.newaxis]

        hidden_error_gradient = np.zeros((self._hidden_size, 1))

        for i in reversed(range(self._sequence_length)):
            hidden_i = self._hidden_state[i]

            if self._return_sequence:
                l_grad_i = e[i]
                d_output_weights += np.matmul(l_grad_i, hidden_i.T)
                d_output_bias += l_grad_i

                hidden_error = np.matmul(self._output_weights.T, l_grad_i) + hidden_error_gradient
            else:
                if i == self._sequence_length - 1:
                    d_output_weights += np.matmul(e, hidden_i.T)
                    d_output_bias += e

                    hidden_error = np.matmul(self._output_weights.T, e)
                else:
                    hidden_error = hidden_error_gradient

            hidden_derivative = self._d_activation_function(hidden_i)
            h_grad_i = hidden_derivative * hidden_error

            if i > 0:
                d_hidden_weights += np.matmul(h_grad_i, self._hidden_state[i - 1].T)
                d_hidden_bias += h_grad_i

            input_x = self._x[i].reshape(self._x[i].shape[0], 1)
            d_input_weights += np.matmul(h_grad_i, input_x.T)

            hidden_error_gradient = np.matmul(self._hidden_weights.T, h_grad_i)

            d_x[i] = np.matmul(self._input_weights.T, h_grad_i)

            self._input_weights -= learning_rate * d_input_weights
            self._hidden_weights -= learning_rate * d_hidden_weights
            self._output_weights -= learning_rate * d_output_weights
            self._hidden_bias -= learning_rate * d_hidden_bias
            self._output_bias -= learning_rate * d_output_bias

            return d_x

    def init(self):
        self._hidden_size = self._input_size
        self._input_weights = np.random.random((self._hidden_size, self._input_size)) * 2 - 1
        self._hidden_weights = np.random.random((self._hidden_size, self._hidden_size)) * 2 - 1
        self._output_weights = np.random.random((self._output_size, self._hidden_size)) * 2 - 1

        self._hidden_bias = np.zeros((self._hidden_size, 1))
        self._output_bias = np.zeros((self._output_size, 1))
