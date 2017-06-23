"""
Module to load the dataset of speckled images from the course.
"""

from dlipr.utils import get_datapath, Dataset, maybe_savefig
import numpy as np
import h5py
import matplotlib.pyplot as plt


def load_data():
    """Load the dataset of images of simulated GISAXS measurements
    (Grazing Incidence Small-Angle X-ray Scattering).
    The dataset contains the speckled (noisy) images along with the underlying
    unspeckled images for training a denoising autoencoder.

    Returns:
        Dataset: Speckled and unspeckled images (20000 train, 5500 test)
    """
    data = Dataset()

    # monkey-patch the plot_examples function
    def monkeypatch_method(cls):
        def decorator(func):
            setattr(cls, func.__name__, func)
            return func
        return decorator

    @monkeypatch_method(Dataset)
    def plot_examples(self, num_examples=10, fname=None):
        """Plot the first examples of speckled and unspeckled images.

        Args:
            num_examples (int, optional): number of examples to plot for each class
            fname (str, optional): filename for saving the plot
        """
        fig, axes = plt.subplots(2, num_examples, figsize=(num_examples, 2))
        for i, X in enumerate((self.train_images, self.train_images_noisy)):
            for j in range(num_examples):
                ax = axes[i, j]
                ax.imshow(X[j, :, :, 0])
                ax.set_xticks([])
                ax.set_yticks([])
        axes[0, 0].set_ylabel('unspeckled')
        axes[1, 0].set_ylabel('speckled')
        maybe_savefig(fig, fname)

    fname = get_datapath('AutoEncoder/data.h5')
    fin = h5py.File(fname)['data']

    def preprocess(A):
        A = np.swapaxes(A, 0, 1)
        A = np.log10(A + 0.01)
        A /= np.max(A, axis=1, keepdims=True)
        return np.reshape(A, (len(A), 64, 64, 1))

    speckle = preprocess(fin['speckle_images'])
    normal = preprocess(fin['normal_images'])

    data.train_images_noisy = speckle[:20000]
    data.test_images_noisy = speckle[20000:]
    data.train_images = normal[:20000]
    data.test_images = normal[:20000]
    return data
