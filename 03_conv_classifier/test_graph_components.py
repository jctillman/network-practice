
import numpy as np
from numpy.random import random, rand

from utils_for_tests import (
    same_generator,
    alt_generator,
    generic_test_module_optimize,
    generic_test_module_derivative_amount)

from loss import (
    mean_squared_loss)

from graph_components import (
    MatrixMult, MatrixAdd,
    Relu, Sigmoid, Exponent,
    Probabilize, Softmax, LeakyRelu,
    Elo)
 
from main import FullyConnectedModel


def test_MatrixMultModel():

    def init(input_shape, output_shape):
        return MatrixMult([input_shape[1], output_shape[1]])

    generic_test_module_optimize(init, mean_squared_loss)
    generic_test_module_derivative_amount(init, mean_squared_loss, shape_generator=alt_generator)


def test_MatrixAddModel():

    def init(input_shape, output_shape):
        return MatrixAdd([input_shape[1]])

    generic_test_module_optimize(init, mean_squared_loss, shape_generator=same_generator)
    generic_test_module_derivative_amount(
        init,
        mean_squared_loss,
        shape_generator=same_generator)


def test_Relu():

    def init(input_shape, output_shape):
        return Relu()

    generic_test_module_derivative_amount(
        init,
        mean_squared_loss,
        shape_generator=same_generator)
    

def test_Sigmoid():

    def init(input_shape, output_size):
        return Sigmoid()

    generic_test_module_derivative_amount(
        init,
        mean_squared_loss,
        shape_generator=same_generator)


def test_Exponent():

    def init(input_shape, output_shape):
        return Exponent()

    generic_test_module_derivative_amount(
        init,
        mean_squared_loss,
        shape_generator=same_generator)

    
def test_LeakyRelu():

    def init(input_shape, output_shape):
        return LeakyRelu()

    generic_test_module_derivative_amount(
        init,
        mean_squared_loss,
        shape_generator=same_generator)

    
def test_Elo():

    def init(input_shape, output_shape):
        return Elo()
    
    generic_test_module_derivative_amount(
        init,
        mean_squared_loss,
        shape_generator=same_generator)

    
def test_Probabilize():

    def init(input_shape, output_shape):
        return Probabilize()
    
    generic_test_module_derivative_amount(
        init,
        mean_squared_loss,
        shape_generator=same_generator)

    
def test_Softmax():

    def init(input_shape, output_shape):
        return Softmax()

    generic_test_module_derivative_amount(
        init,
        mean_squared_loss,
        shape_generator=same_generator)

