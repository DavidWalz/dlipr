"""
Small data set of flower photos.
Classes: daisy, dandelion, roses, sunflowers, tulips

"""

from dlipr.utils import get_datapath, Dataset
import numpy as np


def load_data():
    """Load a small dataset of flower photos (daisies, dandelions, roses, sunflowers, tulips)

    Returns:
        Dataset: flower photos and labels
    """
    fname = get_datapath('flower_photos/flowers_224.npz')
    data = np.load(fname)
    X_train, X_test = np.split(data['X'], [-1000])
    y_train, y_test = np.split(data['y'], [-1000])

    data = Dataset()
    data.classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    data.train_images = X_train
    data.train_labels = y_train
    data.test_images = X_test
    data.test_labels = y_test
    return data
