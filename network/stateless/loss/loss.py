import numpy as np


def applied(loss_fnc, name):

    def applied_loss_fnc(prediction=None, truth=None):
        loss, gradient = mean_squared_loss(
                prediction=prediction[name],
                truth=truth[name])
        return loss, { name: gradient }  
    return applied_loss_fnc

def running_rnn_loss(input_var, output_var, basic_loss):

        def loss(input=None, prediction=None):
                losses = []
                derivs = []
                shape = None
                for t in range(0, len(prediction) - 1):
                        loss, deriv = basic_loss(
                                prediction=prediction[t][output_var],
                                truth=input[input_var][:,t + 1,:]
                        )
                        derivs.append({ output_var: deriv })
                        losses.append(loss)
                        shape = deriv.shape
                derivs = derivs + [{output_var: np.zeros(shape)}]
                return sum(losses), derivs
                
        return loss

def mean_squared_loss(prediction=None, truth=None):
    assert prediction.shape == truth.shape
    loss = 0.5 * np.sum(np.power(prediction - truth, 2))
    deriv = prediction - truth
    return loss, deriv
