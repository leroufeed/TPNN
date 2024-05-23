import pandas as pd
from sklearn.model_selection import train_test_split
from Perceptron import PerceptronModel

from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

data_frame = pd.read_csv("C:\\Users\\laris\\PycharmProjects\\Preprocessing\\.venv\\archive.zip")

data_frame = data_frame.drop("Brand", axis=1)

for name in data_frame.columns:
    max_value = data_frame[name].max()
    min_value = data_frame[name].min()
    data_frame[name] = (data_frame[name] - min_value) / (max_value - min_value)

X = data_frame.drop("Price", axis=1)
y = data_frame["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

perceptron = PerceptronModel([5, 1, 1])


y_predicted = perceptron.predict(X_test)


