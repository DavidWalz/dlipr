"""
Module to load an Ising dataset from the course.
See https://arxiv.org/abs/1605.01735
"""

from dlipr.utils import get_datapath, Dataset
import numpy as np
import pickle


def load_data():
    """Load the Ising data set.

    Returns:
        Dataset: Ising data
    """

    def load_pickle(fname):
        with open(fname, 'rb') as handle:
            return pickle.load(handle)

    fname = get_datapath('Ising/batch_%i.pickle')
    X_train = np.empty((50000, 1024))
    y_train = np.empty((50000))

    for i in range(5):
        X, y = load_pickle(fname % (i + 1))
        X_train[i * 10000: (i + 1) * 10000] = X
        y_train[i * 10000: (i + 1) * 10000] = y

    X_test, y_test = load_pickle(fname % 5)

    data = Dataset()
    data.train_images = X_train
    data.train_labels = y_train
    data.test_images = X_test
    data.test_labels = y_test
    data.classes = np.arange(0.2, 5.1, 0.2)  # temperature
    return data
