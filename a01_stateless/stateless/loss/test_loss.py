
import numpy as np
from numpy.random import random, randint, rand

from stateless.loss.loss import (
    mean_squared_loss)

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

def test_mean_squared_loss():
    generic_test_loss_direction(mean_squared_loss, 4)
    generic_test_loss_amount(mean_squared_loss, 4)

