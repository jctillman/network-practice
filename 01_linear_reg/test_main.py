
from main import MeanSquared
import numpy as np


def test_MeanSquared():

    ms = MeanSquared()

    loss, deriv = ms(
        np.array([[0,1]]),
        np.array([[1,0]]))

    assert loss == 1
    
    loss, deriv = ms(
        np.array([[0,1],[1, 0]]),
        np.array([[1,0],[0, 1]]))

    assert loss == 1
    
    loss, deriv = ms(
        np.array([[0.5,0.5],[0.5, 0.5]]),
        np.array([[1,0],[0, 1]]))
    
    assert loss == 0.25



