"""
Module to load an Ising dataset from the course.
See https://arxiv.org/abs/1605.01735
"""

from dlipr.utils import get_datapath, Dataset
import numpy as np


def load_data():
    """Load the Ising data set.

    Returns:
        Dataset: Ising data
    """
    data = np.load(get_datapath('Ising/data.npz'))
    X = data['C']
    y = data['T']

    temperatures = np.arange(1, 3.51, 0.1)
    y = np.searchsorted(temperatures, y)

    X_train, X_test = np.split(X, [22000])
    y_train, y_test = np.split(y, [22000])

    data = Dataset()
    data.classes = temperatures
    data.train_images = X_train
    data.train_labels = y_train
    data.test_images = X_test
    data.test_labels = y_test
    return data
