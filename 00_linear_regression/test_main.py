import numpy as np
from numpy.random import random, randint, rand

from main import (
    tree_height_generator,
    mean_squared_loss,
    LinearRegressionModel)


def test_tree_height_generator():

    for i in range(2, 5):
        gen = tree_height_generator(i)
        x, y = next(gen)
        assert len(x) == i
        assert len(y) == i

def random_test_matrices():
    batch_size = randint(low=1, high=6)
    indiv = int(randint(low=1,high=size))
    truth = rand(batch_size, size)
    pred = rand(batch_size, size)
    

def generic_test_loss_direction(fnc, size):
    '''
    Just tests that subtracting derivative
    moves it in the correct direction, first
    of all.
    '''
    LR = 0.001
    for __ in range(10):

        batch_size = randint(low=1, high=6)
        truth = rand(batch_size, size)
        pred = rand(batch_size, size)
        old_loss = None

        for _ in range(10):
            loss, deriv = fnc(prediction=pred, truth=truth)
            pred = pred - deriv * 0.001
            assert old_loss is None or loss < old_loss
            old_loss = loss


def generic_test_loss_amount(fnc, size):
    '''
    Tests that altering one element in loss
    scaled by derivative basically alters
    the loss by the right amount
    '''
    for __ in range(100):

        batch_size = randint(low=1, high=6)
        indiv = int(randint(low=1,high=size))
        truth = rand(batch_size, size)
        pred = rand(batch_size, size)
        old_loss = None
        LR = 0.00001
        for _ in range(10):
            loss, deriv = fnc(prediction=pred, truth=truth)
            
            # Algo breaks down where derivative is mostly flat
            if deriv[indiv] < 0.05:
                break
            
            change_amount = (1 / deriv[indiv]) * LR

            for bs in range(batch_size):
                pred[bs][indiv] = pred[bs][indiv] - change_amount

            assert old_loss is None or np.isclose(
                (old_loss - loss) * (1 / LR),
                1,
                atol=0.01)

            old_loss = loss


def generic_test_module_optimize(model_cls, loss_fnc):
    
    for __ in range(10):

        output_size = randint(low=1, high=3)
        input_size = randint(low=2, high=4)
        batch_size = randint(low=2, high=5)
        model = model_cls(input_size, output_size)
        
        inputs = rand(batch_size, input_size)
        truth = rand(batch_size, output_size)
        LR = 0.0001
        old_loss = None

        for _ in range(10):
            outputs = model.forward(inputs)
            loss, derivative = loss_fnc(prediction=outputs, truth=truth)
            model.backward(derivative)
            model.optimize(LR)
            assert old_loss == None or old_loss > loss

def generic_test_module_derivative_direction(model_cls, loss_fnc):
    
    for __ in range(10):

        output_size = randint(low=1, high=3)
        input_size = randint(low=2, high=4)
        batch_size = randint(low=2, high=5)
        model = model_cls(input_size, output_size)
        
        inputs = rand(batch_size, input_size)
        truth = rand(batch_size, output_size)
        LR = 0.0001
        old_loss = None

        for _ in range(10):
            outputs = model.forward(inputs)
            loss, derivative = loss_fnc(prediction=outputs, truth=truth)
            input_derivative = model.backward(derivative)
            inputs = inputs - input_derivative * LR
            assert old_loss == None or old_loss > loss

def generic_test_module_derivative_amount(module_cls, loss_fnc):
    '''
    Tests that altering one element in loss
    scaled by derivative basically alters
    the loss by the right amount
    '''
    for __ in range(100):
        
        output_size = randint(low=1, high=3)
        input_size = randint(low=2, high=4)
        batch_size = randint(low=2, high=5)
        model = module_cls(input_size, output_size)

        indiv = int(randint(low=1,high=input_size))
        inputs = rand(batch_size, input_size)
        truth = rand(batch_size, output_size)
        old_loss = None
        LR = 0.00001
        for _ in range(10):
            
            outputs = model.forward(inputs)
            loss, derivative = loss_fnc(prediction=outputs, truth=truth)
            input_derivative = model.backward(derivative)

            # Algo breaks down where derivative is mostly flat
            if input_derivative[indiv] < 0.05:
                break
            
            change_amount = (1 / input_derivative[indiv]) * LR

            for bs in range(batch_size):
                inputs[bs][indiv] = inputs[bs][indiv] - change_amount

            assert old_loss is None or np.isclose(
                (old_loss - loss) * (1 / LR),
                1,
                atol=0.01)

            old_loss = loss



def test_mean_squared_loss():
    generic_test_loss_direction(mean_squared_loss, 4)
    generic_test_loss_amount(mean_squared_loss, 4)

def test_LRModel():
    generic_test_module_optimize(LinearRegressionModel, mean_squared_loss)
    generic_test_module_derivative_direction(LinearRegressionModel, mean_squared_loss)
    generic_test_module_derivative_amount(LinearRegressionModel, mean_squared_loss)

