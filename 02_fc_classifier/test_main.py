import numpy as np
from numpy.random import random, randint, rand

from main import (
    tree_height_generator,
    mnist_generator,
    mean_squared_loss,
    FullyConnectedModel,
    MatrixMult, MatrixAdd,
    Relu, Sigmoid, Exponent,
    Probabilize, Softmax, LeakyRelu,
    Elo)


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
    tries = 100
    skips = 0
    for batch_size, truth, pred, indiv in loss_test_matrices(tries):

        # First, find old_loss and derivative
        old_loss, deriv = fnc(prediction=pred, truth=truth)
            
        # This testing technique breaks down
        # where derivative is mostly flat
        if any([ deriv[x][indiv] < 0.05 for x in range(deriv.shape[0])]):
            continue
        
        # Change prediction by an amount such that
        # we expect loss to change by LR
        for bs in range(batch_size):
            change_amount = (1 / deriv[bs][indiv]) * LR
            pred[bs][indiv] = pred[bs][indiv] - change_amount
        
        # Find new loss
        new_loss, deriv = fnc(prediction=pred, truth=truth)

        # Difference between old_loss and new_loss
        # times reciprocal of LR should be unity
        print((old_loss - new_loss) * (1 / LR))
        assert np.isclose(
                (old_loss - new_loss) * (1 / LR),
                1 * batch_size,
                atol=0.1)

def module_test_matrices(times, init_fnc, same_size):

    for _ in range(times):
        batch_size = randint(low=1, high=3)
        output_size = randint(low=1, high=3)
        input_size = randint(low=1, high=3)
        if same_size:
            output_size=input_size
        indiv = int(randint(low=0,high=input_size))
        inp = rand(batch_size, input_size) * 0.5 + 0.25
        out = rand(batch_size, output_size) * 0.5 + 0.25
        model = init_fnc(input_size, output_size)
        yield batch_size, model, inp, out, indiv

def module_test_matrices_normalized(times, init_fnc, same_size):

    for _ in range(times):
        batch_size = randint(low=1, high=4)
        output_size = randint(low=1, high=4)
        input_size = randint(low=1, high=4)
        if same_size:
            output_size=input_size
        indiv = int(randint(low=0,high=input_size))
        inp = rand(batch_size, input_size) * 0.5 + 0.25
        out = rand(batch_size, output_size) * 0.5 + 0.25

        out = out / np.expand_dims(np.sum(out, axis=1), axis=1)

        model = init_fnc(input_size, output_size)
        yield batch_size, model, inp, out, indiv 
        
def module_test_matrices_negative(times, init_fnc, same_size):

    for _ in range(times):
        batch_size = randint(low=1, high=4)
        output_size = randint(low=1, high=4)
        input_size = randint(low=1, high=4)
        if same_size:
            output_size=input_size
        indiv = int(randint(low=0,high=input_size))
        inp = rand(batch_size, input_size) - 0.5
        out = rand(batch_size, output_size) - 0.5

        out = out / np.expand_dims(np.sum(out, axis=1), axis=1)

        model = init_fnc(input_size, output_size)
        yield batch_size, model, inp, out, indiv 


def generic_test_module_optimize(init_fnc, loss_fnc, same_size=False):
    
    LR = 0.0001
    tries = 100
    failures = 0
    for _, model, inputs, truth, __ in module_test_matrices(tries, init_fnc, same_size):
        
        outputs = model.forward(inputs)
        old_loss, derivative = loss_fnc(prediction=outputs, truth=truth)
        model.backward(derivative)
        model.optimize(LR)

        new_outputs = model.forward(inputs)
        new_loss, derivative = loss_fnc(prediction=new_outputs, truth=truth)
        if new_loss > old_loss:
            failures = failures + 1
    
    assert failures < tries / 5

def generic_test_module_derivative_direction(
    init_fnc, loss_fnc, matrices=module_test_matrices, same_size=False):
    
    tries = 100
    LR = 0.0001
    skips = 0
    for _, model, inputs, truth, __ in matrices(tries, init_fnc, same_size):
        
        outputs = model.forward(inputs)
        loss, derivative = loss_fnc(prediction=outputs, truth=truth)
        input_derivative = model.backward(derivative)
        inputs = inputs - input_derivative * LR

        # Algo breaks down where derivative is mostly flat
        in_d = input_derivative
        if np.any(np.abs(in_d) < 0.0025):
            skips += 1

        new_outputs = model.forward(inputs)
        new_loss, derivative = loss_fnc(prediction=new_outputs, truth=truth)

        assert new_loss <= loss or (loss == 0)

    assert tries - skips > tries / 10

def generic_test_module_derivative_amount(
    init_fnc, loss_fnc, matrices=module_test_matrices, same_size=False):
    '''
    Tests that altering one element in loss
    scaled by derivative basically alters
    the loss by the right amount
    '''
    tries = 100
    skips = 0
    LR = 0.0000001
    for batch_size, model, inputs, truth, indiv in matrices(tries, init_fnc, same_size):
        
        outputs = model.forward(inputs)
        old_loss, derivative = loss_fnc(prediction=outputs, truth=truth)
        input_deriv = model.backward(derivative)

        # Algo breaks down where derivative is mostly flat
        in_d = input_deriv
        if np.any(np.abs(in_d) < 0.0025):
            skips += 1
            continue

        for bs in range(batch_size):
            change_amount = (1 / input_deriv[bs][indiv]) * LR
            inputs[bs][indiv] = inputs[bs][indiv] - change_amount
        
        new_outputs = model.forward(inputs)
        new_loss, derivative = loss_fnc(prediction=new_outputs, truth=truth)

        assert np.isclose(
            (old_loss - new_loss) * (1 / LR),
            1 * batch_size,
            atol=0.03) or (old_loss == 0.0 and new_loss == 0.0)

    assert tries - skips > tries / 10


def test_mean_squared_loss():
    generic_test_loss_direction(mean_squared_loss, 4)
    generic_test_loss_amount(mean_squared_loss, 4)

def test_FullyConnectedModelModel():

    def init(input_size, output_size):
        return FullyConnectedModel([input_size, output_size])

    generic_test_module_optimize(init, mean_squared_loss)
    generic_test_module_derivative_direction(init, mean_squared_loss)
    generic_test_module_derivative_amount(init, mean_squared_loss)

def test_MatrixMultModel():


    def init(input_size, output_size):
        return MatrixMult([input_size, output_size])

    generic_test_module_optimize(init, mean_squared_loss)
    generic_test_module_derivative_direction(init, mean_squared_loss)
    generic_test_module_derivative_amount(init, mean_squared_loss)

def test_MatrixAddModel():

    def init(input_size, output_size):
        return MatrixAdd([input_size])

    generic_test_module_optimize(init, mean_squared_loss, same_size=True)
    generic_test_module_derivative_direction(init, mean_squared_loss, same_size=True)
    generic_test_module_derivative_amount(init, mean_squared_loss, same_size=True)

def test_Relu():

    def init(input_size, output_size):
        return Relu()

    generic_test_module_derivative_direction(
        init,
        mean_squared_loss,
        same_size=True,
        matrices=module_test_matrices_negative)
    
    generic_test_module_derivative_amount(
        init,
        mean_squared_loss,
        same_size=True,
        matrices=module_test_matrices_negative)

def test_Sigmoid():

    def init(input_size, output_size):
        return Sigmoid()

    generic_test_module_derivative_direction(init, mean_squared_loss, same_size=True)
    generic_test_module_derivative_amount(init, mean_squared_loss, same_size=True)


def test_Exponent():

    def init(input_size, output_size):
        return Exponent()

    generic_test_module_derivative_direction(init, mean_squared_loss, same_size=True)
    generic_test_module_derivative_amount(init, mean_squared_loss, same_size=True)

def test_LeakyRelu():

    def init(input_size, output_size):
        return LeakyRelu()

    generic_test_module_derivative_direction(
        init,
        mean_squared_loss,
        same_size=True,
        matrices=module_test_matrices_negative)
    
    generic_test_module_derivative_amount(
        init,
        mean_squared_loss,
        same_size=True,
        matrices=module_test_matrices_negative)

def test_Elo():

    def init(input_size, output_size):
        return Elo()
    
    generic_test_module_derivative_direction(
        init,
        mean_squared_loss,
        same_size=True,
        matrices=module_test_matrices_negative)
    
    generic_test_module_derivative_amount(
        init,
        mean_squared_loss,
        same_size=True,
        matrices=module_test_matrices_negative)

def test_Probabilize():

    def init(input_size, output_size):
        return Probabilize()

    generic_test_module_derivative_direction(
        init,
        mean_squared_loss,
        same_size=True,
        matrices=module_test_matrices_normalized)
    generic_test_module_derivative_amount(
        init,
        mean_squared_loss,
        same_size=True,
        matrices=module_test_matrices_normalized)

def test_Softmax():

    def init(input_size, output_size):
        return Softmax()

    generic_test_module_derivative_direction(
        init,
        mean_squared_loss,
        same_size=True,
        matrices=module_test_matrices_normalized)
    generic_test_module_derivative_amount(
        init,
        mean_squared_loss,
        same_size=True,
        matrices=module_test_matrices_normalized)
