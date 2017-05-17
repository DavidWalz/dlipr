"""
Module to load the CIFAR-10 and CIFAR-100 datasets of tiny natural images.
See http://www.cs.toronto.edu/~kriz/cifar.html

The CIFAR datasets are labeled subsets of the 80 million tiny images dataset
collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
The images are of size 32x32 pixels with 3 color channels (RGB).

CIFAR-10
    10 classes containing 5000 training and 1000 test images, each.
    airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck


CIFAR-100
    100 classes containing 600 images each (500 training and 100 testing).
    The classes (fine labels) are grouped into 20 superclasses (coarse labels).

Superclass                      Classes
aquatic mammals                 mammals beaver, dolphin, otter, seal, whale
fish                            aquarium fish, flatfish, ray, shark, trout
flowers                         orchids, poppies, roses, sunflowers, tulips
food containers                 bottles, bowls, cans, cups, plates
fruit and vegetables            apples, mushrooms, oranges, pears, sweet peppers
household electrical devices    clock, computer keyboard, lamp, telephone, TV
household furniture             bed, chair, couch, table, wardrobe
insects                         bee, beetle, butterfly, caterpillar, cockroach
large carnivores                bear, leopard, lion, tiger, wolf
large man-made outdoor things   bridge, castle, house, road, skyscraper
large natural outdoor scenes    cloud, forest, mountain, plain, sea
large omnivores and herbivores  camel, cattle, chimpanzee, elephant, kangaroo
medium-sized mammals            fox, porcupine, possum, raccoon, skunk
non-insect invertebrates        crab, lobster, snail, spider, worm
people                          baby, boy, girl, man, woman
reptiles                        crocodile, dinosaur, lizard, snake, turtle
small mammals                   hamster, mouse, rabbit, shrew, squirrel
trees                           maple, oak, palm, pine, willow
vehicles 1                      bicycle, bus, motorcycle, pickup truck, train
vehicles 2                      lawn-mower, rocket, streetcar, tank, tractor
"""

from dlipr.utils import get_datapath, Dataset
import numpy as np

try:
    # python 2
    import cPickle as pickle
    _pickle_kwargs = {}
except ImportError:
    # python 3
    import pickle
    _pickle_kwargs = {'encoding': 'latin1'}


cifar10_labels = np.array([
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'])

cifar100_coarse_labels = np.array([
    'aquatic mammals',
    'fish',
    'flowers',
    'food containers',
    'fruit and vegetables',
    'household electrical devices',
    'household furniture',
    'insects',
    'large carnivores',
    'large man-made outdoor things',
    'large natural outdoor scenes',
    'large omnivores and herbivores',
    'medium-sized mammals',
    'non-insect invertebrates',
    'people',
    'reptiles',
    'small mammals',
    'trees',
    'vehicles 1',
    'vehicles 2'])

cifar100_fine_labels = np.array([
    'beaver', 'dolphin', 'otter', 'seal', 'whale',
    'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
    'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
    'bottles', 'bowls', 'cans', 'cups', 'plates',
    'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
    'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
    'bed', 'chair', 'couch', 'table', 'wardrobe',
    'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
    'bear', 'leopard', 'lion', 'tiger', 'wolf',
    'bridge', 'castle', 'house', 'road', 'skyscraper',
    'cloud', 'forest', 'mountain', 'plain', 'sea',
    'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
    'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
    'crab', 'lobster', 'snail', 'spider', 'worm',
    'baby', 'boy', 'girl', 'man', 'woman',
    'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
    'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
    'maple', 'oak', 'palm', 'pine', 'willow',
    'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
    'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'])


def load_pickle(fname, label_key='labels'):
    """ Load a CIFAR pickle

    Args:
        fname (str): path to pickle file
        label_key (str, optional): labels to retrieve

    Returns: (X, y)
        X (array): images, shape = (N, 32, 32, 3)
        y (array): labels, shape = (N), integers
    """
    with open(fname, mode='rb') as file:
        p = pickle.load(file, **_pickle_kwargs)
        X = p['data']
        y = p[label_key]

    # reshape image data (N, 3*32*32) --> (N, 32, 32, 3)
    X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(y, dtype='uint8')
    return X, y


def load_cifar10():
    """Load the CIFAR-10 data set.

    Returns:
        Dataset: CIFAR-10 data
    """
    print('Loading CIFAR-10 dataset')

    fname = get_datapath('CIFAR/cifar-10-data/batch_%i.pickle')
    X_train = np.empty((50000, 32, 32, 3), dtype='uint8')
    y_train = np.empty((50000), dtype='uint8')

    for i in range(5):
        X, y = load_pickle(fname % (i + 1))
        X_train[i * 10000: (i + 1) * 10000] = X
        y_train[i * 10000: (i + 1) * 10000] = y

    X_test, y_test = load_pickle(fname % 6)

    data = Dataset()
    data.classes = cifar10_labels
    data.train_images = X_train
    data.train_labels = y_train
    data.test_images = X_test
    data.test_labels = y_test
    return data


def load_cifar100(label_key='fine_labels'):
    """Load CIFAR-100 data set using the 'fine_labels' or 'coarse_labels'.

    Returns:
        Dataset: CIFAR-100 data
    """
    print('Loading CIFAR-100 dataset with %s' % label_key)

    fname = get_datapath('CIFAR/cifar-100-data/%s.pickle')
    X_train, y_train = load_pickle(fname % 'train', label_key)
    X_test, y_test = load_pickle(fname % 'test', label_key)

    data = Dataset()

    if label_key == 'fine_labels':
        data.classes = cifar100_fine_labels
        # stored integers refer to alphabetically sorted fine labels
        sorting = np.argsort(data.classes)
        y_train = np.array([sorting[y] for y in y_train], dtype='uint8')
        y_test = np.array([sorting[y] for y in y_test], dtype='uint8')
    else:
        data.classes = cifar100_coarse_labels

    data.train_images = X_train
    data.train_labels = y_train
    data.test_images = X_test
    data.test_labels = y_test
    return data
