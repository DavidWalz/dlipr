"""
Module to load an Ising dataset from the course.
See https://arxiv.org/abs/1605.01735
"""

from dlipr.utils import get_datapath, Dataset
import numpy as np
import pickle


def load_data():
    """Load the Ising data set."""

    def load_pickle(fname):
        with open('data%i.pickle' % i, 'rb') as handle:
            return pickle.load(handle)  # returns (X, y)
    
    fname = get_datapath('Ising/batch_%i.pickle')
    X_train = np.empty((50000, 1024))
    y_train = np.empty((50000))

    for i in range(5):
        X, y = load_pickle(fname % i)
            X_train[i*10000 : (i+1)*10000] = X
            y_train[i*10000 : (i+1)*10000] = y

    X_test, y_test = load_pickle(fname % 5)

    data = Dataset()
    data.classes = np.arange(0.2, 5.1, 0.2)  # temperature
    data.train_images = X_train
    data.train_labels = y_train
    data.test_images = X_test
    data.test_labels = y_test
    return data

def classify_data(y_data, Tc):
    """ assign temperatures to magnetic phases """
    Y_data = np.zeros((len(y_data), 2))
    Y_data[np.where(y_data < Tc), :] = [1, 0]
    Y_data[np.where(y_data > Tc), :] = [0, 1]
    return Y_data