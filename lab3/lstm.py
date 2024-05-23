from numpy.typing import NDArray

from RNN.layer import Layer
import numpy as np
from RNN.activation_functions import sigmoid, d_sigmoid


class LSTM(Layer):
    def __init__(self, output_size: int, activation: str = 'none', input_size=None, return_sequence=False):
        super().__init__(activation, input_shape=input_size)
        self._hidden_size = 0
        self._output_size = output_size
        self._return_sequence = return_sequence
        self._sequence_length = 0

        self.Wi = np.empty(0)
        self.Ui = np.empty(0)
        self.bi = np.empty(0)

        self.Wf = np.empty(0)
        self.Uf = np.empty(0)
        self.bf = np.empty(0)

        self.Wo = np.empty(0)
        self.Uo = np.empty(0)
        self.bo = np.empty(0)

        self.Wc = np.empty(0)
        self.Uc = np.empty(0)
        self.bc = np.empty(0)

        self.Wy = np.empty(0)
        self.by = np.empty(0)

        self.x = np.empty(0)
        self.hs = np.empty(0)
        self.cs = np.empty(0)
        self.os = np.empty(0)
        self.fs = np.empty(0)
        self.ins = np.empty(0)
        self._cs = np.empty(0)
        self.c_tanh = np.empty(0)

    def get_output_shape(self):
        return self._output_size

    def forward(self, x):
        self._sequence_length = x.shape[0]
        self.x = x

        h_k = np.zeros((self._hidden_size, 1))
        c_k = np.zeros((self._hidden_size, 1))

        self.hs = np.zeros((self._sequence_length, self._hidden_size, 1))
        self.cs = np.zeros((self._sequence_length, self._hidden_size, 1))
        self.ins = np.zeros((self._sequence_length, self._hidden_size, 1))
        self.os = np.zeros((self._sequence_length, self._hidden_size, 1))
        self.fs = np.zeros((self._sequence_length, self._hidden_size, 1))
        self._cs = np.zeros((self._sequence_length, self._hidden_size, 1))
        self.c_tanh = np.zeros((self._sequence_length, self._hidden_size, 1))

        for k in range(self._sequence_length):
            x_k = x[k]
            if len(x_k.shape) == 1:
                x_k = x_k[:, np.newaxis]
            ik = sigmoid(
                np.matmul(self.Wi, x_k) +
                np.matmul(self.Ui, h_k) +
                self.bi
            )
            fk = sigmoid(
                np.matmul(self.Wf, x_k) +
                np.matmul(self.Uf, h_k) +
                self.bf
            )
            ok = sigmoid(
                np.matmul(self.Wo, x_k) +
                np.matmul(self.Uo, h_k) +
                self.bo
            )

            c = self._activation_function(
                np.matmul(self.Wc, x_k) +
                np.matmul(self.Uc, h_k) +
                self.bc
            )

            self.ins[k] = ik
            self.os[k] = ok
            self.fs[k] = fk
            self._cs[k] = c

            ck = fk * c_k + ik * c
            self.cs[k] = ck

            self.c_tanh[k] = self._activation_function(ck)
            ht = ok * self.c_tanh[k]
            self.hs[k] = ht

        if self._return_sequence:
            output = (np.matmul(self.Wy, self.hs.T) + self.by).T
        else:
            final_hidden_state = self.hs[-1].reshape(self.hs[-1].shape[0], 1)
            output = np.matmul(self.Wy, final_hidden_state) + self.by

        return output

    def backward(self, e, learning_rate: float):
        dWi, dUi, dbi = np.zeros_like(self.Wi), np.zeros_like(self.Ui), np.zeros_like(self.bi)
        dWf, dUf, dbf = np.zeros_like(self.Wf), np.zeros_like(self.Uf), np.zeros_like(self.bf)
        dWo, dUo, dbo = np.zeros_like(self.Wo), np.zeros_like(self.Uo), np.zeros_like(self.bo)
        dWc, dUc, dbc = np.zeros_like(self.Wc), np.zeros_like(self.Uc), np.zeros_like(self.bc)
        dWy, dby = np.zeros_like(self.Wy), np.zeros_like(self.by)

        dh_next = np.zeros((self._hidden_size, 1))
        dc_next = np.zeros((self._hidden_size, 1))

        dX = np.zeros((self._sequence_length, self._input_size, 1))

        for k in reversed(range(self._sequence_length)):
            if self._return_sequence:
                dy = e[k]
            else:
                if k == self._sequence_length - 1:
                    dy = e
                else:
                    dy = np.zeros_like(e)

            dWy += np.matmul(dy, self.hs[k].T)
            dby += dy
            dh = np.matmul(self.Wy.T, dy) + dh_next
            dc = (self.os[k] * dh *
                  self._d_activation_function(self.cs[k]) + dc_next)
            dot = d_sigmoid(self.os[k]) * self.c_tanh[k] * dh

            if k > 0:
                dft = self.cs[k - 1] * dc * d_sigmoid(self.fs[k])
            else:
                dft = np.zeros_like(self.fs[k])

            dit = self._cs[k] * dc * d_sigmoid(self.ins[k])
            dct = self.ins[k] * dc * self._d_activation_function(self._cs[k])
            dWi += np.matmul(self.x[k].T, dit)
            dbi += dit

            dWf += np.matmul(self.x[k].T, dft)
            dbf += dft

            dWo += np.matmul(self.x[k].T, dot)
            dbo += dot

            dWc += np.matmul(self.x[k].T, dct)
            dbc += dct

            if k > 0:
                dUi += np.matmul(self.hs[k - 1].T, dit)
                dUf += np.matmul(self.hs[k - 1].T, dft)
                dUo += np.matmul(self.hs[k - 1].T, dot)
                dUc += np.matmul(self.hs[k - 1].T, dct)

            dh_next = (
                    np.matmul(self.Ui.T, dit) +
                    np.matmul(self.Uf.T, dft) +
                    np.matmul(self.Uo.T, dot) +
                    np.matmul(self.Uc.T, dct)
            )
            dc_next = self.fs[k] * dc

            dX[k] = (
                    np.matmul(self.Wi.T, dit) +
                    np.matmul(self.Wf.T, dft) +
                    np.matmul(self.Wo.T, dot) +
                    np.matmul(self.Wc.T, dct)
            )

        self.Wi -= learning_rate * dWi
        self.Ui -= learning_rate * dUi
        self.bi -= learning_rate * dbi
        self.Wf -= learning_rate * dWf
        self.Uf -= learning_rate * dUf
        self.bf -= learning_rate * dbf
        self.Wo -= learning_rate * dWo
        self.Uo -= learning_rate * dUo
        self.bo -= learning_rate * dbo
        self.Wc -= learning_rate * dWc
        self.Uc -= learning_rate * dUc
        self.bc -= learning_rate * dbc
        self.Wy -= learning_rate * dWy
        self.by -= learning_rate * dby

        return dX

    def init(self):
        self._hidden_size = self._input_size

        self.Wi = np.random.random((self._hidden_size, self._input_size)) * 2 - 1
        self.Ui = np.random.random((self._hidden_size, self._hidden_size)) * 2 - 1
        self.bi = np.zeros((self._hidden_size, 1))

        self.Wf = np.random.random((self._hidden_size, self._input_size)) * 2 - 1
        self.Uf = np.random.random((self._hidden_size, self._hidden_size)) * 2 - 1
        self.bf = np.zeros((self._hidden_size, 1))

        self.Wo = np.random.random((self._hidden_size, self._input_size)) * 2 - 1
        self.Uo = np.random.random((self._hidden_size, self._hidden_size)) * 2 - 1
        self.bo = np.zeros((self._hidden_size, 1))

        self.Wc = np.random.random((self._hidden_size, self._input_size)) * 2 - 1
        self.Uc = np.random.random((self._hidden_size, self._hidden_size)) * 2 - 1
        self.bc = np.zeros((self._hidden_size, 1))

        self.Wy = np.random.random((self._output_size, self._hidden_size)) * 2 - 1
        self.by = np.zeros((self._output_size, 1))
