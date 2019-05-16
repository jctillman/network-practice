import numpy as np
from numpy.random import random, randint, rand

from datagen import (
    tree_height_generator,
    mnist_generator)

def test_mnist_generator():
    for i in range(2, 5):
        gen = mnist_generator(i)
        x, y = next(gen)
        assert len(x) == i
        assert len(x[0] == 28)
        assert len(x[0] == 28)
        assert np.max(x) < 0.51

def test_tree_height_generator():

    for i in range(2, 5):
        gen = tree_height_generator(i)
        x, y = next(gen)
        assert len(x) == i
        assert len(y) == i
