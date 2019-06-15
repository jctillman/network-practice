import numpy as np
from math import floor
from random import random

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

all_linkers = [
    Relu,
    Exponent,
    Identity,
    Probabilize,
    Concat,
    MatrixAdd,
    MatrixMult,
    MatrixAddExact]


def test_all_backprob_again():
    
    i = Identity([], name='input')
    iw = Identity([], name='fc_w1')
    ib = Identity([], name='fc_b1')

    h1 = Sigmoid([MatrixAdd([MatrixMult([i, iw], name='mult1'), ib], name='add1')], name='h1')
    iw2 = Identity([], name='fc_w2')
    ib2 = Identity([], name='fc_b2')
    h2 = Sigmoid([MatrixAdd([MatrixMult([h1, iw2]), ib2])], name='h2')
    h3 = MatrixAddExact([h1, h2], name='added')

    iw3 = Identity([], name='fc_w3')
    ib3 = Identity([], name='fc_b3')

    h4 = Relu(MatrixAdd([MatrixMult([h3, iw3]), ib3]))

    h5 = MatrixAddExact([i, h4])

    output = Probabilize(Exponent(h4, name='output'))

    full = output
    
    rand = np.random.rand

    def input_generator():
        return {
            'input': rand(*[7, 10]),
            'fc_w1': rand(*[10, 11]),
            'fc_b1': rand(*[11]),
            'fc_w2': rand(*[11, 11]),
            'fc_b2': rand(*[11]),
            'fc_w3': rand(*[11, 10]),
            'fc_b3': rand(*[10]),
        }

    skips = 0
    for n in range(100):

        inpp = input_generator()
        desired = rand(*[7, 10])

        forward1 = full.forw(inpp)

        loss1, deriv1 = mean_squared_loss(
            prediction=forward1['output'],
            truth=desired)

        derivatives = full.back(
            { 'output': deriv1 },
            forward1,
            list(inpp.keys()))

        k = list(derivatives.keys())
        r = np.random.choice(k)

        indiv = inpp[r].copy()

        random_point = [ floor(i * random()) for i in indiv.shape]

        this_deriv = derivatives[r]
        for ii in range(len(random_point)):
            cord = random_point[ii]
            this_deriv = this_deriv[cord]

        if np.abs(this_deriv) < 0.001:
            skips += 1
            continue

        LR = 0.001
        change_amount = LR
        if len(random_point) == 1:
            indiv[random_point[0]] = indiv[random_point[0]] - change_amount
        elif len(random_point) == 2:
            indiv[random_point[0]][random_point[1]] = indiv[random_point[0]][random_point[1]] - change_amount
        elif len(random_point) == 3:
            indiv[random_point[0]][random_point[1]][random_point[2]] = indiv[random_point[0]][random_point[1]][random_point[2]] - change_amount
        elif len(random_point) == 4:
            indiv[random_point[0]][random_point[1]][random_point[2]][random_point[3]] = indiv[random_point[0]][random_point[1]][random_point[2]][random_point[3]] - change_amount
        else:
            assert False


        inpp[r] = indiv
        forward2 = full.forw(inpp)
        loss2, deriv2 = mean_squared_loss(
            prediction=forward2['output'],
            truth=desired)

        amount = (loss1 - loss2)
        
        assert np.isclose(loss1, loss2, this_deriv * LR, atol=0.01) or (loss1 == 0.0 and loss2  == 0.0)


    assert skips < 50

def test_all_backprob():
    
    i = Identity([], name='input')
    iw = Identity([], name='fc_w1')
    ib = Identity([], name='fc_b1')

    h1 = Sigmoid([MatrixAdd([MatrixMult([i, iw], name='mult1'), ib], name='add1')], name='h1')
    iw2 = Identity([], name='fc_w2')
    ib2 = Identity([], name='fc_b2')
    h2 = Sigmoid([MatrixAdd([MatrixMult([h1, iw2]), ib2])], name='h2')

    h3 = MatrixAddExact([h1, h2], name='added')

    output = Exponent(h3, name='output')

    full = output
    
    rand = np.random.rand

    def input_generator():
        return {
            'input': rand(*[7, 10]),
            'fc_w1': rand(*[10, 11]),
            'fc_b1': rand(*[11]),
            'fc_w2': rand(*[11, 11]),
            'fc_b2': rand(*[11]),
        }

    skips = 0
    for n in range(100):

        inpp = input_generator()
        desired = rand(*[7, 11])

        forward1 = full.forw(inpp)

        loss1, deriv1 = mean_squared_loss(
            prediction=forward1['output'],
            truth=desired)

        derivatives = full.back(
            { 'output': deriv1 },
            forward1,
            list(inpp.keys()))

        k = list(derivatives.keys())
        r = np.random.choice(k)

        indiv = inpp[r].copy()

        random_point = [ floor(i * random()) for i in indiv.shape]

        this_deriv = derivatives[r]
        for ii in range(len(random_point)):
            cord = random_point[ii]
            this_deriv = this_deriv[cord]

        if np.abs(this_deriv) < 0.001:
            skips += 1
            continue

        LR = 0.001
        change_amount = LR
        if len(random_point) == 1:
            indiv[random_point[0]] = indiv[random_point[0]] - change_amount
        elif len(random_point) == 2:
            indiv[random_point[0]][random_point[1]] = indiv[random_point[0]][random_point[1]] - change_amount
        elif len(random_point) == 3:
            indiv[random_point[0]][random_point[1]][random_point[2]] = indiv[random_point[0]][random_point[1]][random_point[2]] - change_amount
        elif len(random_point) == 4:
            indiv[random_point[0]][random_point[1]][random_point[2]][random_point[3]] = indiv[random_point[0]][random_point[1]][random_point[2]][random_point[3]] - change_amount
        else:
            assert False


        inpp[r] = indiv
        forward2 = full.forw(inpp)
        loss2, deriv2 = mean_squared_loss(
            prediction=forward2['output'],
            truth=desired)

        amount = (loss1 - loss2)
        
        assert np.isclose(loss1, loss2, this_deriv * LR, atol=0.01) or (loss1 == 0.0 and loss2  == 0.0)


    assert skips < 50










def test_names_manual():
    
    i = Identity([], name='input')
    iw = Identity([], name='fc_w1')
    ib = Identity([], name='fc_b1')

    h1 = Relu([MatrixAdd([MatrixMult([i, iw], name='mult1'), ib], name='add1')], name='h1')
    iw2 = Identity([], name='fc_w2')
    ib2 = Identity([], name='fc_b2')
    h2 = Relu([MatrixAdd([MatrixMult([h1, iw2]), ib2])], name='h2')
    output = Probabilize(Exponent(h2))

    full = output

    includes = full.get_names()
    should_include = [
        'input', 'fc_w1', 'fc_b1',
        'h1', 'fc_b2', 'fc_w2', 'h2']

    for name in should_include:
        assert name in includes

    predecessors = full.get_inputs_required_for(['h1'])
    assert len(predecessors) == 3
    should_include = ['input', 'fc_w1', 'fc_b1']
    for name in should_include:
        assert name in includes

    requires_input = full.get_inputs()
    assert len(requires_input) == 5
    

    for _ in range(200):
        
        i = np.random.rand(10, 21)
        w1 = np.random.rand(21, 13)
        b1 = np.random.rand(13)
        
        i_dict = { 'input': i, 'fc_w1': w1, 'fc_b1': b1 }
        results = full.forw(i_dict, [ 'h1' ])
        desired_h1 = np.random.rand(*results['h1'].shape)

        old_loss, deriv = mean_squared_loss(
            prediction=results['h1'],
            truth=desired_h1)

        for __ in range(2):
            
            i_dict = { 'input': i, 'fc_w1': w1, 'fc_b1': b1 }
            results = full.forw(i_dict, [ 'h1' ])

            loss, deriv = mean_squared_loss(
                prediction=results['h1'],
                truth=desired_h1)

            back_derivs = full.back(
                { 'h1': deriv },
                results,
                [ 'fc_w1', 'fc_b1'])

            w1 = w1 - back_derivs['fc_w1'] * 0.001
            b1 = b1 - back_derivs['fc_b1'] * 0.001

        i_dict = { 'input': i, 'fc_w1': w1, 'fc_b1': b1 }
        results = full.forw(i_dict, [ 'h1' ])

        new_loss, deriv = mean_squared_loss(
            prediction=results['h1'],
            truth=desired_h1)

        assert new_loss < old_loss





