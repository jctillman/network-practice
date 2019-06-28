import numpy as np


def applied(loss_fnc, name):

    def applied_loss_fnc(prediction=None, truth=None):
        loss, gradient = mean_squared_loss(
                prediction=prediction[name],
                truth=truth[name])
        return loss, { name: gradient }  
    return applied_loss_fnc

def mean_squared_loss(prediction=None, truth=None):
    
    assert prediction.shape == truth.shape
    loss = 0.5 * np.sum(np.power(prediction - truth, 2))
    deriv = prediction - truth
    return loss, deriv
