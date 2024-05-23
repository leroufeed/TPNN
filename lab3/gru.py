from numpy.typing import NDArray

from RNN.layer import Layer
import numpy as np
from RNN.activation_functions import sigmoid, d_sigmoid


class GRU(Layer):
    def __init__(self, output_size: int, activation: str = 'none', input_size=None, return_sequence=False):
        super().__init__(activation, input_shape=input_size)
        self._hidden_size = 0
        self._output_size = output_size
        self._return_sequence = return_sequence
        self._sequence_length = 0

        self.Wz = np.empty(0)
        self.Uz = np.empty(0)
        self.bz = np.empty(0)

        self.Wr = np.empty(0)
        self.Ur = np.empty(0)
        self.br = np.empty(0)

        self.Wh = np.empty(0)
        self.Uh = np.empty(0)
        self.bh = np.empty(0)

        self.Wy = np.empty(0)
        self.by = np.empty(0)

        self.x = np.empty(0)
        self.z = np.empty(0)
        self.r = np.empty(0)
        self.h = np.empty(0)
        self.h_hat = np.empty(0)

    def get_output_shape(self):
        return self._output_size

    def forward(self, x):
        self._sequence_length = x.shape[0]
        self.x = x

        z = np.zeros((self._sequence_length, self._hidden_size, 1))
        r = np.zeros((self._sequence_length, self._hidden_size, 1))
        h = np.zeros((self._sequence_length, self._hidden_size, 1))
        h_hat = np.zeros((self._sequence_length, self._hidden_size, 1))

        if len(x.shape) == 2:
            x = x[:, :, np.newaxis]

        for k in range(self._sequence_length):

            if k > 0:
                prev_h = h[k - 1]
            else:
                prev_h = np.zeros_like(h[0])

            z[k] = sigmoid(
                np.matmul(self.Wz, x[k]) +
                np.matmul(self.Uz, prev_h) +
                self.bz
            )

            r[k] = sigmoid(
                np.matmul(self.Wr, x[k]) +
                np.matmul(self.Ur, prev_h) +
                self.br
            )

            h_hat[k] = self._activation_function(
                np.matmul(self.Wh, x[k]) +
                np.matmul(self.Uh, prev_h) +
                self.bh
            )

            h[k] = z[k] * prev_h + (1 - z[k]) * h_hat[k]

        self.z = z
        self.r = r
        self.h = h
        self.h_hat = h_hat

        if self._return_sequence:
            return np.matmul(self.Wy, self.h) + self.by
        else:
            return np.matmul(self.Wy, self.h[-1]) + self.by

    def backward(self, e, learning_rate: float):
        dWz, dUz, dbz = np.zeros_like(self.Wz), np.zeros_like(self.Uz), np.zeros_like(self.bz)
        dWr, dUr, dbr = np.zeros_like(self.Wr), np.zeros_like(self.Ur), np.zeros_like(self.br)
        dWh, dUh, dbh = np.zeros_like(self.Wh), np.zeros_like(self.Uh), np.zeros_like(self.bh)
        dWy, dby = np.zeros_like(self.Wy), np.zeros_like(self.by)

        dh_next = np.zeros((self._hidden_size, 1))

        dX = np.zeros((self._sequence_length, self._input_size, 1))
        if len(self.x.shape) == 2:
            self.x = self.x[:, :, np.newaxis]

        for k in reversed(range(self._sequence_length)):
            if self._return_sequence:
                dy = e[k]
            else:
                if k == self._sequence_length - 1:
                    dy = e
                else:
                    dy = np.zeros_like(e)

            dWy += np.matmul(dy, self.h[k].T)
            dby += dy

            dh = np.matmul(self.Wy.T, dy) + dh_next
            dh_hat = dh * (1 - self.z[k])
            dh_hat_l = dh_hat * self._d_activation_function(self.h_hat[k])

            dWh += np.matmul(dh_hat_l, self.x[k].T)
            dUh += np.matmul(dh_hat_l, (self.r[k] * self.h[k - 1]).T)
            dbh += dh_hat_l

            drhp = np.matmul(self.Uh.T, dh_hat_l)
            dr = drhp * self.h[k - 1]
            dr_l = dr * self._d_activation_function(self.r[k])

            dWr += np.matmul(dr_l, self.x[k].T)
            dUr += np.matmul(dr_l, self.h[k - 1].T)
            dbr += dr_l

            dz = dh * (self.h[k - 1] - self.h_hat[k])
            dz_l = dz * self._d_activation_function(self.z[k])

            dWz += np.matmul(dz_l, self.x[k].T)
            dUz += np.matmul(dz_l, self.h[k - 1].T)
            dbz += dz_l

            dh_fz_inner = np.matmul(self.Uz.T, dz_l)
            dh_fz = dh * self.z[k]
            dh_fhh = drhp * self.r[k]
            dh_fr = np.matmul(self.Ur.T, dr_l)

            dh_next = dh_fz_inner + dh_fz + dh_fhh + dh_fr

            dX[k] = (np.matmul(self.Wz.T, dz_l) +
                     np.matmul(self.Wr.T, dr_l) +
                     np.matmul(self.Wh.T, dh_hat_l))

        self.Wz -= learning_rate * dWz
        self.Uz -= learning_rate * dUz
        self.bz -= learning_rate * dbz
        self.Wr -= learning_rate * dWr
        self.Ur -= learning_rate * dUr
        self.br -= learning_rate * dbr
        self.Wh -= learning_rate * dWh
        self.Uh -= learning_rate * dUh
        self.bh -= learning_rate * dbh
        self.Wy -= learning_rate * dWy
        self.by -= learning_rate * dby

        return dX

    def init(self):
        self._hidden_size = self._input_size

        self.Wz = np.random.random((self._hidden_size, self._input_size)) - 0.5
        self.Uz = np.random.random((self._hidden_size, self._hidden_size)) - 0.5
        self.bz = np.zeros((self._hidden_size, 1))

        self.Wr = np.random.random((self._hidden_size, self._input_size)) - 0.5
        self.Ur = np.random.random((self._hidden_size, self._hidden_size)) - 0.5
        self.br = np.zeros((self._hidden_size, 1))

        self.Wh = np.random.random((self._hidden_size, self._input_size)) - 0.5
        self.Uh = np.random.random((self._hidden_size, self._hidden_size)) - 0.5
        self.bh = np.zeros((self._hidden_size, 1))

        self.Wy = np.random.random((self._output_size, self._hidden_size)) - 0.5
        self.by = np.zeros((self._output_size, 1))
