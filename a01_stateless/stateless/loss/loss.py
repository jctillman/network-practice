import numpy as np

def mean_squared_loss(prediction=None, truth=None):
    
    assert prediction.shape == truth.shape
    loss = 0.5 * np.sum(np.power(prediction - truth, 2))
    deriv = prediction - truth
    return loss, deriv
