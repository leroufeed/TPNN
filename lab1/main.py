import pandas as pd
from HeatMap import show_heatmap, show_lib_heatmap
from GainRatio import gain_ratio
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

data_frame = pd.read_csv("C:\\Users\\laris\\PycharmProjects\\Preprocessing\\.venv\\archive.zip")

show_heatmap(data_frame)

data_frame = data_frame.drop("Brand", axis=1)

n = 5
for name in data_frame.columns:
    unique_count = len(data_frame[name].unique())
    if unique_count > 5:
        data_frame["D" + name] = pd.qcut(data_frame[name], q=n, labels=range(n))
        data_frame = data_frame.drop(name, axis=1)

target = "DPrice"

for name in data_frame:
    gr = gain_ratio(data_frame, target, name)
    print(f"{name} = {gr}")


X = data_frame.drop(target, axis=1)
y = data_frame[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

decisionTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)

decisionTree.fit(X_train, y_train)
y_predicted = decisionTree.predict(X_test)
print(accuracy_score(y_predicted, y_test))


fig = plt.figure(figsize=(25, 25))
_ = tree.plot_tree(decisionTree, feature_names=X.columns, filled=True)
plt.show()
