import numpy as np


class PerceptronModel:
    def __init__(self, layers_sizes):
        self.layers = []
        self.layers_count = len(layers_sizes) - 1
        self.layers_sizes = layers_sizes
        for i in range(len(layers_sizes) - 1):
            self.layers.append(np.random.rand(layers_sizes[i + 1], layers_sizes[i] + 1) - 0.5)

        self.sums = []
        for i in range(1, len(self.layers_sizes)):
            self.sums.append(np.zeros((self.layers_sizes[i], 1)))

        self.layers_result = []
        for i in range(len(self.layers_sizes)):
            self.layers_result.append(np.zeros((self.layers_sizes[i], 1)))

        self.errors = []
        for i in range(1, len(self.layers_sizes)):
            self.errors.append(np.zeros((self.layers_sizes[i], 1)))

    def fit(self, X_test_data_frame, y_test_data_frame):
        X_test = np.array(X_test_data_frame)
        y_test = np.array(y_test_data_frame)

        err = 100
        while err > 0.01 * len(y_test):
            err = 0
            for i in range(len(y_test)):
                result = self.forward_propagation(X_test[i])
                err += abs((y_test[i] - result)[0][0])
                self.errors[self.layers_count - 1] = np.array(y_test[i] - result)

                for i in range(self.layers_count - 2, -1, -1):
                    self.errors[i] = np.dot(self.layers[i + 1].T, self.errors[i + 1])
                    self.errors[i] = self.errors[i][0:len(self.errors[i]) - 1]

                for i in range(self.layers_count - 1):
                    res = np.repeat(self.layers_result[i].T, self.layers[i].shape[0], axis=0)
                    self.sums[i][self.sums[i] >= 0] = 1
                    self.sums[i][self.sums[i] < 0] = 0
                    f_diff_sum = np.repeat(self.sums[i] * self.errors[i], self.layers[i].shape[1], axis=1)
                    self.layers[i] = self.layers[i] + (f_diff_sum * res) * 0.001

            print(err / len(y_test))

    def forward_propagation(self, input_vector):
        result = np.concatenate([input_vector, [1]])
        result = np.array(result)
        self.layers_result[0] = result.reshape((len(result), 1))
        for i in range(self.layers_count - 1):
            self.layers_result[i + 1] = np.dot(self.layers[i], self.layers_result[i])
            self.sums[i] = np.copy(self.layers_result[i + 1])
            self.layers_result[i + 1][self.layers_result[i + 1] < 0] = 0
            self.layers_result[i + 1] = np.expand_dims(np.concatenate([self.layers_result[i + 1].T[0], [1]]).T, axis=1)

        i = self.layers_count - 1
        self.layers_result[i + 1] = np.dot(self.layers[i], self.layers_result[i])
        self.sums[i] = np.copy(self.layers_result[i + 1])

        return self.layers_result[i + 1]

    def predict(self, input_vectors):
        out = []
        for elem in np.array(input_vectors):
            result = np.concatenate([elem, [1]])
            for i in range(self.layers_count - 1):
                result = np.dot(self.layers[i], result)
                result[result < 0] = 0
                result = np.concatenate([result, [1]])

            result = np.dot(self.layers[self.layers_count - 1], result)[0]

            out.append(result)

        return np.array(out).T


if __name__ == "__main__":
    model = PerceptronModel([3, 3, 1])
    model.fit([[1, 2, 3], [-1, 5, 1], [0, 0, 0]], [1, 5, -1])
    print(model.predict([1, 2, 3]))
    print(model.predict([-1, 5, 1]))
    print(model.predict([0, 0, 0]))
