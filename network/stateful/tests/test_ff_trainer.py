import pytest

from stateless.utils.dag import Dag
from stateless.utils.utils import list_equals
from stateless.loss.loss import mean_squared_loss, applied
from stateless.optimizer.sgd import get_sgd_optimizer
from stateful.ff_trainer import FFTrainer
from numpy.random import rand

from datagen.datagen import tree_height_generator, mnist_generator, mapper


from stateless.graph.graph_linked import (
    Relu,
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

def test_works_simple():

    i = Input('input')
    iw = Parameter('fc_w1')
    ib = Parameter('fc_b1')
    h1 = Sigmoid([MatrixAdd([MatrixMult([i, iw], name='mult1'), ib], name='add1')], name='h1')
    iw2 = Parameter('fc_w2')
    ib2 = Parameter('fc_b2')
    h2 = Relu(MatrixAdd([MatrixMult([h1, iw2]), ib2]))
    output = Probabilize(Exponent(h2, name='output'))

    weights = {
        'fc_w1': rand(*[3, 11]),
        'fc_b1': rand(*[11]),
        'fc_w2': rand(*[11, 1]),
        'fc_b2': rand(*[1]),
    }

    batch_size = 10
    generator = mapper(tree_height_generator, 10, 'input', 'output')

    trainer = FFTrainer(
        output,
        weights,
        applied(mean_squared_loss, 'output'),
        get_sgd_optimizer(0.001))

    xs, ys = generator()
    loss_before = trainer.test_single(xs, ys)
    trainer.train_batch(100, generator)
    loss_after = trainer.test_single(xs, ys)
    assert loss_after * 10 < loss_before

def test_works_mnist():

    i = Input('input')
    iw = Parameter('fc_w1')
    ib = Parameter('fc_b1')
    h1 = Sigmoid([MatrixAdd([MatrixMult([i, iw], name='mult1'), ib], name='add1')], name='h1')
    iw2 = Parameter('fc_w2')
    ib2 = Parameter('fc_b2')
    h2 = Relu(MatrixAdd([MatrixMult([h1, iw2]), ib2]))
    output = Probabilize(Exponent(h2, name='output'))

    weights = {
        'fc_w1': rand(*[28 * 28, 30]),
        'fc_b1': rand(*[30]),
        'fc_w2': rand(*[30, 10]),
        'fc_b2': rand(*[10]),
    }

    generator = mapper(mnist_generator, 2, 'input', 'output')

    trainer = FFTrainer(
        output,
        weights,
        applied(mean_squared_loss, 'output'),
        get_sgd_optimizer(0.001))

    
    xs, ys = generator()
    loss_before = trainer.test_single(xs, ys)
    trainer.train_batch(200, generator)
    loss_after = trainer.test_single(xs, ys)

    assert loss_after * 2 < loss_before



