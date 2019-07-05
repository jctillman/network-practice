
import numpy as np
from math import floor


from stateless.graph.graph_linked import (
    Relu,
    Exponent,
    Identity,
    Probabilize,
    Concat,
    Sigmoid,
    MatrixAdd,
    MatrixMult,
    MatrixAddExact)

from stateless.loss.loss import mean_squared_loss
from stateless.graph.rnn_converter import to_rnn
from datagen.datagen import stupid_fsm

def test_basic_rnn():
    
    i = Identity([], name='input')
    fcw1 = Identity([], name='fc_w1')
    fcb1 = Identity([], name='fc_b1')

    ii = Identity([], name='prior_h1')
    joined = Concat([i, ii])
    h1 = Relu(MatrixAdd([MatrixMult([joined, fcw1]), fcb1]), name='h1')

    fcw2 = Identity([], name='fc_w2')
    fcb2 = Identity([], name='fc_b2')

    output = (
        Relu(
            MatrixAdd([MatrixMult([h1, fcw2]), fcb2]),
        name='output')
    )

    BN = 4
    T = 15
    NUM = 3    
    rnn = to_rnn(output)

    H_SIZE = 5
    weights = {
        'fc_w1': 0.05 * np.random.rand(3 + H_SIZE, H_SIZE),
        'fc_b1': 0.05 * np.random.rand(H_SIZE),
        'fc_w2': 0.05 * np.random.rand(H_SIZE, NUM),
        'fc_b2': 0.05 * np.random.rand(NUM),
    }

    for i in range(50):

        forward_data = stupid_fsm()

        forward = rnn.forw(
            { 'input': forward_data },
            weights,
            { 'h1': np.zeros((BN, H_SIZE))}
        )

        losses = []
        derivs = []
        for ii in range(0, T):
            loss, deriv = mean_squared_loss(
                prediction=forward[ii]['output'],
                truth=forward_data[:,ii,:]
            )
            derivs.append({ 'output': deriv })
            losses.append(loss)
        
        print("Loss at ", i, " is ", sum(losses))

        backwards = rnn.back(
            forward,
            derivs,
            ['fc_w1', 'fc_w2', 'fc_b1', 'fc_b2', 'h1']
        )

        for key in weights.keys():
            weights[key] = weights[key] - 0.01 * backwards[key]
