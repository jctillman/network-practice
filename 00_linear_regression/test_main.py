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

def loss_test_matrices(times):

    for _ in range(times):
        batch_size = randint(low=1, high=6)
        size = randint(low=1, high=6)
        truth = rand(batch_size, size)
        pred = rand(batch_size, size)
        indiv = randint(low=0, high=size)
        yield batch_size, truth, pred, indiv
    

def generic_test_loss_direction(fnc, size):
    '''
    Just tests that subtracting derivative
    moves it in the correct direction, first
    of all.
    '''
    LR = 0.001
    for _, truth, pred, __ in loss_test_matrices(20):
        
        old_loss, deriv = fnc(prediction=pred, truth=truth)
        pred = pred - deriv * 0.001
        new_loss, deriv = fnc(prediction=pred, truth=truth)
        assert new_loss < old_loss


def generic_test_loss_amount(fnc, size):
    '''
    Tests that altering one element in loss
    scaled by derivative basically alters
    the loss by the right amount
    '''
    LR = 0.00001
    for batch_size, truth, pred, indiv in loss_test_matrices(100):

        # First, find old_loss and derivative
        old_loss, deriv = fnc(prediction=pred, truth=truth)
            
        # This testing technique breaks down
        # where derivative is mostly flat
        if deriv[indiv] < 0.05:
            continue
        
        # Change prediction by an amount such that
        # we expect loss to change by LR
        change_amount = (1 / deriv[indiv]) * LR
        for bs in range(batch_size):
            pred[bs][indiv] = pred[bs][indiv] - change_amount
        
        # Find new loss
        new_loss, deriv = fnc(prediction=pred, truth=truth)

        # Difference between old_loss and new_loss
        # times reciprocal of LR should be unity
        assert np.isclose(
                (old_loss - new_loss) * (1 / LR),
                1,
                atol=0.01)


def module_test_matrices(times, model_cls):

    for _ in range(times):
        batch_size = randint(low=1, high=6)
        output_size = randint(low=1, high=3)
        input_size = randint(low=2, high=4)
        indiv = int(randint(low=1,high=input_size))
        inp = rand(batch_size, input_size)
        out = rand(batch_size, output_size)
        model = model_cls(input_size, output_size)
        yield batch_size, model, inp, out, indiv 


def generic_test_module_optimize(model_cls, loss_fnc):
    
    LR = 0.0001
    tries = 100
    failures = 0
    for _, model, inputs, truth, __ in module_test_matrices(tries, model_cls):
        
        outputs = model.forward(inputs)
        old_loss, derivative = loss_fnc(prediction=outputs, truth=truth)
        model.backward(derivative)
        model.optimize(LR)

        new_outputs = model.forward(inputs)
        new_loss, derivative = loss_fnc(prediction=new_outputs, truth=truth)
        if new_loss > old_loss:
            failures = failures + 1
    
    assert failures < tries / 10

def generic_test_module_derivative_direction(model_cls, loss_fnc):
    
    tries = 100
    LR = 0.0001
    for _, model, inputs, truth, __ in module_test_matrices(tries, model_cls):
        
        outputs = model.forward(inputs)
        loss, derivative = loss_fnc(prediction=outputs, truth=truth)
        input_derivative = model.backward(derivative)
        inputs = inputs - input_derivative * LR

        new_outputs = model.forward(inputs)
        new_loss, derivative = loss_fnc(prediction=new_outputs, truth=truth)
        assert new_loss < loss

def generic_test_module_derivative_amount(module_cls, loss_fnc):
    '''
    Tests that altering one element in loss
    scaled by derivative basically alters
    the loss by the right amount
    '''
    tries = 100
    LR = 0.00001
    for batch_size, model, inputs, truth, indiv in module_test_matrices(tries, module_cls):
        
        outputs = model.forward(inputs)
        old_loss, derivative = loss_fnc(prediction=outputs, truth=truth)
        input_derivative = model.backward(derivative)

        # Algo breaks down where derivative is mostly flat
        if input_derivative[indiv] < 0.1:
            continue
        
        change_amount = (1 / input_derivative[indiv]) * LR

        for bs in range(batch_size):
            inputs[bs][indiv] = inputs[bs][indiv] - change_amount
        
        new_outputs = model.forward(inputs)
        new_loss, derivative = loss_fnc(prediction=new_outputs, truth=truth)

        assert np.isclose(
            (old_loss - new_loss) * (1 / LR),
            1,
            atol=0.01)




def test_mean_squared_loss():
    generic_test_loss_direction(mean_squared_loss, 4)
    generic_test_loss_amount(mean_squared_loss, 4)

def test_LRModel():
    generic_test_module_optimize(LinearRegressionModel, mean_squared_loss)
    generic_test_module_derivative_direction(LinearRegressionModel, mean_squared_loss)
    generic_test_module_derivative_amount(LinearRegressionModel, mean_squared_loss)

