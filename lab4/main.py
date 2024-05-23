import numpy as np

from convolution_neural_network import ConvolutionNeuralNetwork
from convolution_layer import ConvolutionLayer
from pooling_layer import PoolingLayer
from dense_layer import DenseLayer
from flatten_layer import FlattenLayer
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    model = ConvolutionNeuralNetwork()

    #model.add_layer(ConvolutionLayer(filters_count=6, kernel_size=(5, 5), activation='relu', input_shape=(1, 28, 28)))
    #model.add_layer(PoolingLayer(size=2))
    #model.add_layer(ConvolutionLayer(filters_count=16, kernel_size=(5, 5), activation='relu'))
    #model.add_layer(PoolingLayer(size=2))
    #model.add_layer(FlattenLayer())
    #model.add_layer(DenseLayer(neurons_count=120, activation='relu', input_shape=5))
    #model.add_layer(DenseLayer(neurons_count=84, activation='relu'))
    #model.add_layer(DenseLayer(neurons_count=10))
    #model.add_layer(ConvolutionLayer(filters_count=6, kernel_size=(2, 2), activation='relu', input_shape=(1, 4, 4)))
    #model.add_layer(PoolingLayer)
    #model.add_layer(FlattenLayer())
    #model.add_layer(DenseLayer(neurons_count=5))

    model.add_layer(DenseLayer(neurons_count=6, activation='relu', input_shape=2))
    model.add_layer(DenseLayer(neurons_count=1))

    model.init()

    """

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    y_train = np.eye(10)[y_train][:, np.newaxis]
    y_train = np.transpose(y_train, (0, 2, 1))
    y_test = np.eye(10)[y_test][:, np.newaxis]

    X_train = X_train[:100]
    y_train = y_train[:100]

    """

    X_train = np.array([
        [[0], [0]],
        [[0], [1]],
        [[1], [0]],
        [[1], [1]]])
    y_train = np.array([[0, 0, 0, 0]]).T

    while True:
        error = 0
        for i in range(len(X_train)):
            error += model.fit(np.array(X_train[i]), y_train[i])
            model.update_weight(0.00000001)
        error /= len(X_train)

        print(error)
        if error < 0.1:
            break

