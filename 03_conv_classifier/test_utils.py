import numpy as np
from numpy.random import random, randint, rand

from utils import to_patches_2d, from_patches_2d


def test_patches():

    # Test that for a conv with receptive
    # range of 1, kernel size 1, we just
    # get identity transformations
    for x in range(5, 11):
        for y in range(5, 11):

            patch_size = np.random.choice([1, 3, 5], 1)[0]
            inp_chan = randint(low=1, high=4)

            inp = rand(x, y, inp_chan)
            patched = to_patches_2d(inp, size=patch_size, stride=1)
            same = from_patches_2d(patched, x, y, inp_chan, size=patch_size, reconstruct=True)
            
            assert np.array_equal(inp, same)
