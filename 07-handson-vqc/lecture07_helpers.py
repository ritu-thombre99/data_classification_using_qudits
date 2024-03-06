from pennylane import numpy as np
import matplotlib.pyplot as plt


def load_hands_on_data():
    train_X = np.load("data/l7_train_X.npy")
    train_y = np.load("data/l7_train_y.npy")
    test_X = np.load("data/l7_test_X.npy")
    test_y = np.load("data/l7_test_y.npy")
    
    train_X = np.array(train_X, requires_grad=False)
    train_y = np.array(train_y, requires_grad=False)
    test_X = np.array(test_X, requires_grad=False)
    test_y = np.array(test_y, requires_grad=False)

    return train_X, train_y, test_X, test_y


def plot_data(data, labels):
    """Plot the data and colour by class.
    
    Args:
        data (array[float]): Input data. A list with shape N x 2
            representing points on a 2D plane.
        labels (array[int]): Integers identifying the class/label of 
            each data point.
    """
    plt.scatter(data[:, 0], data[:, 1], c=labels)