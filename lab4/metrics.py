import numpy as np
import matplotlib.pyplot as plt


class ConfusionMatrix:
    def __init__(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0


def print_metrics(y_predicted, y_test):
    confusing_matrix_array = []
    precision_sum = 0
    recall_sum = 0
    TP_sum = 0
    FP_sum = 0
    FN_sum = 0
    for i in range(len(y_test[0])):
        confusing_matrix = get_confusing_matrix(y_predicted, y_test, i)
        confusing_matrix_array.append(confusing_matrix)
        precision = calc_precision(confusing_matrix)
        recall = calc_recall(confusing_matrix)
        f1 = (2 * precision * recall) / (precision + recall)
        precision_sum += precision
        recall_sum += recall
        TP_sum += confusing_matrix.TP
        FP_sum += confusing_matrix.FP
        FN_sum += confusing_matrix.FN
        print(f"{i}: precision: {precision} recall: {recall} f1: {f1}")
    print(f"macro precision: {precision_sum / len(confusing_matrix_array)}")
    print(f"macro recall: {recall_sum / len(confusing_matrix_array)}")
    print(f"micro precision: {TP_sum / (TP_sum + FP_sum)}")
    print(f"micro recall: {TP_sum / (TP_sum + FN_sum)}")



def roc_curve(y_predicted, y_test, target):
    answer = np.concatenate((y_predicted[:, target, :], y_test[:, target, :]), axis=1)
    answer = np.array(sorted(answer, key=lambda x: x[0])).T
    x = [0]
    y = [0]
    for i in range(len(answer[0])):
        if answer[1][i] < 0.5:
            x.append(x[-1])
            y.append(y[-1] + 1)
        else:
            x.append(x[-1] + 1)
            y.append(y[-1])

    plt.plot(x, y, linewidth=0.2, marker='o', markersize=1)


def get_confusing_matrix(y_predicted, y_test, target):
    matrix = ConfusionMatrix()
    for i in range(len(y_test)):
        if y_test[i].argmax() == target and y_predicted[i].argmax() == target:
            matrix.TP += 1
        elif y_test[i].argmax() != target and y_predicted[i].argmax() != target:
            matrix.TN += 1
        elif y_test[i].argmax() == target and y_predicted[i].argmax() != target:
            matrix.FN += 1
        elif y_test[i].argmax() != target and y_predicted[i].argmax() == target:
            matrix.FP += 1

    return matrix


def calc_precision(matrix):
    return matrix.TP / (matrix.TP + matrix.FP)


def calc_recall(matrix):
    return matrix.TP / (matrix.TP + matrix.FN)
