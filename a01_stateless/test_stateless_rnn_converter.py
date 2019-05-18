
import numpy as np
from math import floor
from random import random

from graph_linker import (
    Relu,
    Exponent,
    Identity,
    Probabilize,
    Concat,
    Sigmoid,
    MatrixAdd,
    MatrixMult,
    MatrixAddExact)

from loss import mean_squared_loss

from stateless_rnn_converter import to_rnn

def get_forward():
    BN = 4
    T = 15
    NUM = 3
    forward_data = np.zeros((BN, T, NUM))
    for i in range(BN):
        prior = 'A' if random() > 0.5 else 'C'
        for ii in range(T):
            if prior == 'A':
                if random() > 0.3:
                    next = 'B'
                else:
                    next = 'C'
            if prior == 'B':
                if random() > 0.3:
                    next = 'C'
                else:
                    next = 'B'
            if prior == 'C':
                if random() > 0.5:
                    next = 'A'
                else:
                    next = 'B'
            if next == 'A':
                forward_data[i,ii, 0] = 1
            if next == 'B':
                forward_data[i, ii, 1] = 1
            if next == 'C':
                forward_data[i, ii, 2] = 1
            prior = next
    return forward_data

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

    def data_equality(n1, n2):
        return (
            n1['sgc'] == n2['sgc'] and
            len(n1['input_names']) == len(n2['input_names']) and 
            all([
                x == y for x, y
                in zip(n1['input_names'], n2['input_names'])
            ])
        )

    forward_data = get_forward()
    BN = 4
    T = 15
    NUM = 3    
    rnn = to_rnn(output)

    H_SIZE = 5
    def get_weights():
        return {
            'fc_w1': 0.05 * np.random.rand(3 + H_SIZE, H_SIZE),
            'fc_b1': 0.05 * np.random.rand(H_SIZE),
            'fc_w2': 0.05 * np.random.rand(H_SIZE, NUM),
            'fc_b2': 0.05 * np.random.rand(NUM),
        }

    weights = get_weights()

    for i in range(50):
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

    assert False