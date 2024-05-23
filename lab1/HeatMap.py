import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def show_heatmap(data_frame: pd.DataFrame):
    data_frame = data_frame.drop("Brand", axis=1)

    columns = list(data_frame.columns)

    corr_matrix = __calc_corr_matrix(data_frame)

    __show_matrix_as_heatmap(columns, corr_matrix)


def __calc_corr_matrix(data_frame: pd.DataFrame):
    length = len(data_frame.columns)
    columns_name = list(data_frame.columns)
    corr_matrix = np.zeros((length, length))

    for i in range(length):
        for j in range(length):
            corr_matrix[i][j] = __calc_corr(data_frame[columns_name[i]], data_frame[columns_name[j]])

    return corr_matrix


def __calc_corr(first: pd.Series, second: pd.Series):
    first_mean = first.mean()
    second_mean = second.mean()

    covar = 0
    for i in range(len(first)):
        covar += (first[i] - first_mean) * (second[i] - second_mean)

    return round(covar / (first.std() * second.std() * len(first)), 3)


def __show_matrix_as_heatmap(columns: list, matrix: np.array):
    fig, ax = plt.subplots()
    ax.imshow(matrix)

    ax.set_xticks(np.arange(len(columns)), labels=columns)
    ax.set_yticks(np.arange(len(columns)), labels=columns)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="w")

    fig.tight_layout()
    plt.show()


def show_lib_heatmap(data_frame: pd.DataFrame):
    data_frame = data_frame.drop("Brand", axis=1)
    sns.heatmap(data_frame.corr(), annot=True)
    plt.show()
