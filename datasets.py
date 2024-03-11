import pennylane as qml
from pennylane import numpy as np

from sklearn.model_selection import train_test_split
import numpy

def get_xor_data(size):
    rng = numpy.random.RandomState(0)
    X = np.array(rng.randn(size, 2),requires_grad=False)
    Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    Y = np.array(np.where(Y, 1, -1), requires_grad=False)

    train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.10, random_state=42)
    return train_X, test_X, train_y, test_y