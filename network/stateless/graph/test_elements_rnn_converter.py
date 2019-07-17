
import numpy as np
from math import floor


from stateless.graph.graph_linked import (
    LeakyRelu,
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
from datagen.datagen import stupid_fsm, alt_patterns

def test_basic_rnn():
    
    i = Identity([], name='input')
    fcw1 = Identity([], name='fc_w1')
    fcb1 = Identity([], name='fc_b1')

    ii = Identity([], name='prior_h1')
    joined = Concat([i, ii])
    h1 = LeakyRelu(MatrixAdd([MatrixMult([joined, fcw1]), fcb1]), name='internal_h1')
    h11 = Identity([h1], name='h1')

    fcw2 = Identity([], name='fc_w2')
    fcb2 = Identity([], name='fc_b2')
    i2 = Identity([], name='prior_h2')
    joined2 = Concat([h1, i2])
    h2 = LeakyRelu(MatrixAdd([MatrixMult([joined2, fcw2]), fcb2]), name='internal_h2')
    h22 = Identity([h2], name='h2')

    fcw3 = Identity([], name='fc_w3')
    fcb3 = Identity([], name='fc_b3')

    output = (
        LeakyRelu(
            MatrixAdd([MatrixMult([h2, fcw3]), fcb3]),
        name='output')
    )

    BN = 4
    T = 15
    NUM = 3    
    rnn = to_rnn(output)

    H_SIZE = 13
    weights = {
        'fc_w1': 0.2 * (np.random.rand(3 + H_SIZE, H_SIZE) - 0.5),
        'fc_b1': 0.2 * (np.random.rand(H_SIZE) - 0.5),
        'fc_w2': 0.2 * (np.random.rand(H_SIZE + H_SIZE, H_SIZE) - 0.5),
        'fc_b2': 0.2 * (np.random.rand(H_SIZE) - 0.5),
        'fc_w3': 0.2 * (np.random.rand(H_SIZE, NUM) - 0.5),
        'fc_b3': 0.2 * (np.random.rand(NUM) - 0.5),
    }

    first_loss = None
    last_loss = None
    for i in range(500):

        forward_data = alt_patterns()

        forward = rnn.forw(
            { 'input': forward_data },
            weights,
            { 'h1': np.zeros((BN, H_SIZE)), 'h2': np.zeros((BN, H_SIZE)) }
        )

        losses = []
        derivs = []
        for ii in range(0, T - 1):
            loss, deriv = mean_squared_loss(
                prediction=forward[ii]['output'],
                truth=forward_data[:,ii + 1,:]
            )
            derivs.append({ 'output': deriv })
            losses.append(loss)
        derivs.append({
            'output': np.zeros(derivs[0]['output'].shape)
        })
        
        print("Loss at ", i, " is ", sum(losses))
        if (first_loss is None):
            first_loss = sum(losses)
        last_loss = sum(losses)

        backwards = rnn.back(
            forward,
            derivs,
            ['fc_w1', 'fc_w2', 'fc_w3', 'fc_b1', 'fc_b2', 'fc_b3', 'prior_h1', 'prior_h2']
        )

        for key in weights.keys():
            weights[key] = weights[key] - 0.01 * backwards[key]

    assert last_loss * 20 < first_loss 