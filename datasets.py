import pennylane as qml
from pennylane import numpy as np

from sklearn.model_selection import train_test_split
import numpy
import sklearn
from sklearn import datasets
from ucimlrepo import fetch_ucirepo 

def get_xor_data(size):
    rng = numpy.random.RandomState(0)
    X = np.array(rng.randn(size, 2),requires_grad=False)
    Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    Y = np.array(np.where(Y, 1, -1), requires_grad=False)

    train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.10, random_state=42)
    return train_X, test_X, train_y, test_y


def get_moon_dataset(size):
    moon = sklearn.datasets.make_moons(n_samples=size,noise=0.07)
    X = np.array(moon[0],requires_grad=False)
    y = np.array(np.where(moon[1], 1, -1), requires_grad=False)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.10, random_state=42)
    return train_X, test_X, train_y, test_y


def get_circular_boundary_dataset(size):
    dataset = sklearn.datasets.make_circles(n_samples=size, noise=0.19, factor=0.3)
    X = np.array(dataset[0],requires_grad=False)
    y = np.array(np.where(dataset[1], 1, -1), requires_grad=False)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.10, random_state=42)
    return train_X, test_X, train_y, test_y


def iris_dataset():
    iris = datasets.load_iris()
    X = np.array(iris['data'],requires_grad=False)
    y = np.array(iris['target'],requires_grad=False)
    y = np.where(y == 0, -2, (np.where(y == 1, 0, 2))) #  reformat labels to -2,0,2
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.10, random_state=42) 
    return train_X, test_X, train_y, test_y


def get_wine_dataset():
    wine = fetch_ucirepo(id=109)  
    X = np.array(wine.data.features ,requires_grad=False)
    y = np.array(wine.data.targets,requires_grad=False)
    y = np.where(y == 0, -2, (np.where(y == 1, 0, 2))) #  reformat labels to -2,0,2
    # metadata 
    # print(wine.metadata) 
    # variable information 
    # print(wine.variables)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.10, random_state=42)
    return train_X, test_X, train_y, test_y
  
