import pytest

from stateless.utils.dag import Dag
from stateless.utils.utils import list_equals
from stateless.loss.loss import mean_squared_loss, applied, running_rnn_loss
from stateless.optimizer.sgd import get_sgd_optimizer
from stateful.rnn_trainer import RNNTrainer
from numpy.random import rand
import numpy as np

from datagen.datagen import tree_height_generator, mnist_generator, mapper
from datagen.datagen import stupid_fsm, alt_patterns


from stateless.graph.graph_linked import (
    Relu,
    LeakyRelu,
    Exponent,
    Identity,
    Input,
    Parameter,
    Probabilize,
    Concat,
    Sigmoid,
    MatrixAdd,
    MatrixMult,
    MatrixAddExact)

from stateless.loss.loss import mean_squared_loss

all_linkers = [
    Relu,
    Exponent,
    Identity,
    Input,
    Parameter,
    Probabilize,
    Concat,
    MatrixAdd,
    MatrixMult,
    MatrixAddExact]

def test_rnn_works_simple():

    i = Identity([], name='input')
    fcw1 = Identity([], name='fc_w1')
    fcb1 = Identity([], name='fc_b1')

    ii = Identity([], name='prior_h1')
    joined = Concat([i, ii])
    h1 = Relu(MatrixAdd([MatrixMult([joined, fcw1]), fcb1]), name='h1_internal')

    h11 = Identity([h1], 'h1')


    fcw2 = Identity([], name='fc_w2')
    fcb2 = Identity([], name='fc_b2')

    output = (
        Relu(
            MatrixAdd([MatrixMult([h1, fcw2]), fcb2]),
        name='output')
    )

    BN = 4
    NUM = 3    
    H_SIZE = 6
    weights = {
        'fc_w1': 0.05 * np.random.rand(3 + H_SIZE, H_SIZE),
        'fc_b1': 0.05 * np.random.rand(H_SIZE),
        'fc_w2': 0.05 * np.random.rand(H_SIZE, NUM),
        'fc_b2': 0.05 * np.random.rand(NUM),
    }
    optimizer = get_sgd_optimizer(0.001)
    trainer = RNNTrainer(
        output, 
        weights,
        { 'h1': np.zeros((BN, H_SIZE))},
        running_rnn_loss('input', 'output', mean_squared_loss),
        optimizer)

    def batch_gen():
        return { 'input': stupid_fsm() }

    trainer.train_batch(50, batch_gen)

    trainer.initial_hidden = { 'h1': np.zeros((1, H_SIZE)) }

    num = 20
    initial = np.array([[1,0,0]])
    
    def concretizer(val):
        m = np.random.choice(np.array([0, 1, 2]), p=val['output'][0] / sum(val['output'][0]))
        ret = np.array([0,0,0])
        ret[m] = 1
        return { **val, 'input': np.array([ret]) }

    predicted = trainer.predict(num, {
        'input': initial
    }, concretizer)

def test_rnn_multistep():
    
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
        Exponent(
            MatrixAdd([MatrixMult([h2, fcw3]), fcb3]), name='output')
    )

    BN = 4
    T = 15
    NUM = 3    
    H_SIZE = 13
    weights = {
        'fc_w1': 0.2 * (np.random.rand(3 + H_SIZE, H_SIZE) - 0.5),
        'fc_b1': 0.2 * (np.random.rand(H_SIZE) - 0.5),
        'fc_w2': 0.2 * (np.random.rand(H_SIZE + H_SIZE, H_SIZE) - 0.5),
        'fc_b2': 0.2 * (np.random.rand(H_SIZE) - 0.5),
        'fc_w3': 0.2 * (np.random.rand(H_SIZE, NUM) - 0.5),
        'fc_b3': 0.2 * (np.random.rand(NUM) - 0.5),
    }

    optimizer = get_sgd_optimizer(0.01)
    trainer = RNNTrainer(
        output, 
        weights,
        { 'h1': np.zeros((BN, H_SIZE)), 'h2': np.zeros((BN, H_SIZE)) },
        running_rnn_loss('input', 'output', mean_squared_loss),
        optimizer)
    
    def batch_gen():
        nmn = alt_patterns()
        return { 'input': nmn }

    trainer.train_batch(50, batch_gen)

    trainer.initial_hidden = { 'h1': np.zeros((1, H_SIZE)), 'h2': np.zeros((1, H_SIZE)) }

    num = 20
    initial = np.array([[1,0,0]])
    
    def concretizer(val):
        #m = np.random.choice(np.array([0, 1, 2]), p=val['output'][0] / sum(val['output'][0]))
        print(val['output'])
        m = np.argmax(val['output'])
        ret = np.array([0,0,0])
        ret[m] = 1
        return { **val, 'input': np.array([ret]) }

    predicted = trainer.predict(num, {
        'input': initial
    }, concretizer)

    print([x['input'] for x in predicted])






