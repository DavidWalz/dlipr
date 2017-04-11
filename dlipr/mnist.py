"""
Module to load the MNIST database of handwritten digits
See http://yann.lecun.com/exdb/mnist/

The images are 28x28 pixels (grayscale) showing a single handwritten digit from
0 to 9. The dataset contains 60000 training and 10000 test images.
"""

from dlipr.utils import get_datapath, Dataset
import numpy as np
import gzip


def load_data():
    """Load the MNIST dataset.

    Returns:
        Dataset: MNIST data
    """

    def _read32(bytestream):
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dt)[0]

    def _extract_images(fname):
        with gzip.GzipFile(fileobj=open(fname, 'rb')) as bytestream:
            _read32(bytestream)
            num_images = _read32(bytestream)
            rows = _read32(bytestream)
            cols = _read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            return data.reshape(num_images, rows, cols)

    def _extract_labels(fname):
        with gzip.GzipFile(fileobj=open(fname, 'rb')) as bytestream:
            _read32(bytestream)
            num_items = _read32(bytestream)
            buf = bytestream.read(num_items)
            return np.frombuffer(buf, dtype=np.uint8)

    data = Dataset()
    data.train_images = _extract_images(get_datapath('MNIST/train-images-idx3-ubyte.gz'))
    data.train_labels = _extract_labels(get_datapath('MNIST/train-labels-idx1-ubyte.gz'))
    data.test_images = _extract_images(get_datapath('MNIST/t10k-images-idx3-ubyte.gz'))
    data.test_labels = _extract_labels(get_datapath('MNIST/t10k-labels-idx1-ubyte.gz'))
    data.classes = np.arange(10)
    return data
