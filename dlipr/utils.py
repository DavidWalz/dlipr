"""
Common tools
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os


def get_datapath(fname=''):
    """ Get data path """
    cdir = os.path.dirname(__file__)
    with open(os.path.join(cdir, '.env')) as handle:
        ddir = json.load(handle)['DATA_PATH']
        return os.path.join(ddir, fname)


class Dataset():
    """ Simple dataset container """
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def plot_examples(self, num_examples):
        plot(self, num_examples)


def to_onehot(y, num_classes=None):
    """ Convert integer class labels to one hot encodings, e.g. 2 --> (0,0,1...) """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    onehot = np.zeros((n, num_classes))
    onehot[np.arange(n), y] = 1
    return onehot


def to_label(y):
    """ Converts one hot encodings to integer class labels, e.g. (0,0,1...) --> 2 """
    return np.argmax(y, axis=-1)


def plot_image(X, ax=None):
    """ Plot an image X. """
    if ax is None:
        ax = plt.gca()

    if (X.ndim == 2) or (X.shape[-1] == 1):
        ax.imshow(X.astype('uint8'), origin='upper', cmap=plt.cm.Greys)
    else:
        ax.imshow(X.astype('uint8'), origin='upper')

    ax.set(xticks=[], yticks=[])


def plot_examples(data, num_examples=5):
    """ Plot first examples for each class in given Dataset. """
    num_classes = len(data.classes)
    fig, axes = plt.subplots(num_examples, num_classes, figsize=(num_classes, num_examples))
    for l in range(num_classes):
        axes[0, l].set_title(data.classes[l], fontsize='smaller')
        images = data.train_images[data.train_labels == l]
        for i in range(num_examples):
            plot_image(images[i], axes[i,l])
    return fig


def plot_prediction(X, y, yp, classes, top_n=False):
    """ Plot image along with all or the top_n predictions.

    Args:
        X (array): image
        y (integer): true class label
        yp (array): predicted class scores
        classes (array): class names
        top_n (int, optional): number of top predictions to show
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3.2))
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.15, top=0.98, wspace=0.02)
    plot_image(X, ax1)

    if top_n:
        n = top_n
        s = np.argsort(yp)[-top_n:]
    else:
        n = len(yp)
        s = np.arange(n)[::-1]

    patches = ax2.barh(np.arange(n), yp[s], align='center')
    ax2.set(xlim=(0,1), xlabel='Score', yticks=[])

    for iy, patch in zip(s, patches):
        if iy == y:
            patch.set_facecolor('C1')  # color correct patch

    for i in range(n):
        ax2.text(0.05, i, classes[s][i], ha='left', va='center')

    return fig


# def plot_confusion(y1_true, y1_predict):
#     """ Plot confusion matrix for given true and predicted class labels """
#     C = np.histogram2d(y1_true, y1_predict, bins=np.linspace(-0.5, 19.5, 21))[0]
#     Cn = C / np.sum(C, axis=1)
#     fig = plt.figure(figsize=(12, 12))
#     plt.imshow(Cn, interpolation='nearest', vmin=0, vmax=1, cmap=plt.cm.YlGnBu)
#     plt.colorbar()
#     plt.xlabel('Prediction')
#     plt.ylabel('Truth')
#     plt.xticks(range(20), coarse_labels, rotation='vertical')
#     plt.yticks(range(20), coarse_labels)
#     for x in range(20):
#         for y in range(20):
#             plt.annotate('%i' % C[x,y], xy=(y, x), ha='center', va='center')
#     return fig