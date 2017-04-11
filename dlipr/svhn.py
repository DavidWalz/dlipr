"""
Module to load the Street View House Numbers (SVHN) dataset.
See http://ufldl.stanford.edu/housenumbers/

SVHN is a real-world image dataset for developing machine learning and object
recognition algorithms with minimal requirement on data preprocessing and
formatting. It can be seen as similar in flavor to MNIST (e.g., the images are
of small cropped digits), but incorporates an order of magnitude more labeled
data (over 600,000 digit images) and comes from a significantly harder,
unsolved, real world problem (recognizing digits and numbers in natural scene
images).

SVHN is obtained from house numbers in Google Street View images.
The images are 32x32 pixels centered around a single character (many of the
images do contain some distractors at the sides).
"""

from dlipr.utils import get_datapath, Dataset
import numpy as np
import scipy.io as sio


def load_data(extra=False):
    """Load the SVHN dataset (optionally with extra images)

    Args:
        extra (bool, optional): load extra training data

    Returns:
        Dataset: SVHN data
    """
    def load_mat(fname):
        data = sio.loadmat(fname)
        X = data['X'].transpose(3, 0, 1, 2)
        y = data['y'] % 10  # map label "10" --> "0"
        return X, y

    data = Dataset()
    data.classes = np.arange(10)

    fname = get_datapath('SVHN/%s_32x32.mat')

    X, y = load_mat(fname % 'train')
    data.train_images = X
    data.train_labels = y.reshape(-1)

    X, y = load_mat(fname % 'test')
    data.test_images = X
    data.test_labels = y.reshape(-1)

    if extra:
        X, y = load_mat(fname % 'extra')
        data.extra_images = X
        data.extra_labels = y.reshape(-1)

    return data
