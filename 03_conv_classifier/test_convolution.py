
from numpy.random import random, randint, rand

import numpy as np
from convolution import ConvolutionMult, Convolution, MaxPool
from loss import mean_squared_loss

from utils_for_tests import (
    same_generator,
    alt_generator,
    generic_test_module_optimize,
    generic_test_module_derivative_amount)

def generic_test_Convolution_optimize(conv_cls):

    def shape_generator():
        return (
            [[4,12,12,2],[0,0,0,0]],
            [[4,10,10,2],[0,0,0,0]]
        )

    def init_fnc(input_shape, output_shape):
        return conv_cls(input_shape[1:], 3, output_shape[-1])

    generic_test_module_optimize(init_fnc, mean_squared_loss, shape_generator)
    

def generic_test_Convolution_derivative_amount(conv_cls):

    def shape_generator():
        return (
            [[1,12,12,2],[0,0,0,0]],
            [[1,10,10,3],[0,0,0,0]]
        )

    def init_fnc(input_shape, output_shape):
        return conv_cls(input_shape[1:], 3, output_shape[-1])

    generic_test_module_derivative_amount(init_fnc, mean_squared_loss, shape_generator)


def test_Convolution_vanilla():
    generic_test_Convolution_optimize(ConvolutionMult)
    generic_test_Convolution_derivative_amount(ConvolutionMult)


def test_Convolution():
    generic_test_Convolution_optimize(Convolution)
    generic_test_Convolution_derivative_amount(Convolution)


def generic_test_Pool_derivative_amount(pool):

    def shape_generator():
        return (
            [[1,12,12,2],[0,0,0,0]],
            [[1,6,6,2],[0,0,0,0]]
        )

    def init_fnc(input_shape, output_shape):
        return pool(2)

    generic_test_module_derivative_amount(init_fnc, mean_squared_loss, shape_generator, ratio_skips=0.05)


def test_MaxPool():

    # Super basic testing
    inp = np.array([[[[1],[2]],[[3],[4]]]])
    m = MaxPool(2)
    res = m.forward(inp)
    assert res.reshape(-1) == 4
    m = MaxPool(1)
    assert np.array_equal(m.forward(inp), inp)

    generic_test_Pool_derivative_amount(MaxPool)


