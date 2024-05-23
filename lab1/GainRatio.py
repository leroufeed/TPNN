from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from math import log2


def gain_ratio(data: pd.DataFrame, target: str, feature: str) -> float:
    start_entropy = calc_entropy(data[target])

    unique = data[feature].unique()
    data_len = len(data[target])
    sum_entropy = 0
    split_information = 0
    for i in range(len(unique)):
        sub_data = data[data[feature] == unique[i]][target]
        weight = len(sub_data) / data_len
        sum_entropy += calc_entropy(sub_data)
        split_information -= weight * log2(weight)

    information_gain = start_entropy * len(unique) - sum_entropy
    return information_gain / split_information


def calc_entropy(data: pd.Series) -> float:
    value_counts = data.value_counts()
    count = len(data)
    entropy = 0
    for elem in value_counts:
        freq = elem / count
        if freq > 0.0001:
            entropy -= freq * log2(freq)

    return entropy
