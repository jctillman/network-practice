
import numpy as np
from numpy.random import random, rand

def random_shaped(base, variance):
    shape = [
        x + int(y * random()) for x, y in
        zip(base, variance)]
    return rand(*shape), shape


def data_gen(num, shape_generator):
    
    input_shapes, output_shapes = shape_generator()

    inp_base_shape, inp_variance = input_shapes

    for _ in range(num):
        inp, inp_shape = random_shaped(inp_base_shape, inp_variance)
        out, out_shape = None, None
        if output_shapes is None:
            out, out_shape = random_shaped(inp_shape, [ 0 for _ in inp_shape ])
        else:
            out_base_shape, out_variance = output_shapes
            out, out_shape = random_shaped(out_base_shape, out_variance)
        
        random_points = [
            [ int(x * random()) for x in shape ]
            for shape in [inp_shape, out_shape]
        ]
        yield inp, out, random_points


def default_generator():
    return [[3, 3],[0,3]], [[3, 3],[0,3]]

def same_generator():
    return [[3, 3],[3,3]], None

def alt_generator():
    return [[4, 4],[0,0]], [[4, 4,], [0,0]] 

def generic_test_module_optimize(init_fnc, loss_fnc, shape_generator=default_generator):
    
    LR = 0.0001
    tries = 100
    failures = 0
    
    for inputs, truth, _ in data_gen(tries, shape_generator):
        model = init_fnc(inputs.shape, truth.shape)
        outputs = model.forward(inputs)
        old_loss, derivative = loss_fnc(prediction=outputs, truth=truth)
        model.backward(derivative)
        model.optimize(LR)

        new_outputs = model.forward(inputs)
        new_loss, derivative = loss_fnc(prediction=new_outputs, truth=truth)
        if new_loss > old_loss:
            failures = failures + 1
    
    assert failures < tries / 10


def generic_test_module_derivative_amount(
    init_fnc, loss_fnc, shape_generator=default_generator,
    ratio_skips = 0.5):
    '''
    Tests that altering one element in loss
    scaled by derivative basically alters
    the loss by the right amount
    '''
    tries = 500
    skips = 0
    LR = 0.00000001

    for inputs, truth, random_points in data_gen(tries, shape_generator):

        model = init_fnc(inputs.shape, truth.shape)
        batch_size = inputs.shape[0]
        indiv = random_points[0]
        
        outputs = model.forward(inputs)
        old_loss, derivative = loss_fnc(prediction=outputs, truth=truth)
        input_deriv = model.backward(derivative)

        this_deriv = input_deriv
        for i in range(len(indiv)):
            this_deriv = this_deriv[indiv[i]]

        # Algo breaks down where derivative is mostly flat
        if np.abs(this_deriv) < 0.001:
            skips += 1
            continue

        change_amount = (1 / this_deriv) * LR
        if len(indiv) == 1:
            inputs[indiv[0]] = inputs[indiv[0]] - change_amount
        if len(indiv) == 2:
            inputs[indiv[0]][indiv[1]] = inputs[indiv[0]][indiv[1]] - change_amount
        if len(indiv) == 3:
            inputs[indiv[0]][indiv[1]][indiv[2]] = inputs[indiv[0]][indiv[1]][indiv[2]] - change_amount
        if len(indiv) == 4:
            inputs[indiv[0]][indiv[1]][indiv[2]][indiv[3]] = inputs[indiv[0]][indiv[1]][indiv[2]][indiv[3]] - change_amount
        
        new_outputs = model.forward(inputs)
        new_loss, derivative = loss_fnc(prediction=new_outputs, truth=truth)

        assert np.isclose(
            (old_loss - new_loss) * (1 / LR),
            1,
            atol=0.01) or (old_loss == 0.0 and new_loss == 0.0)

    assert tries - skips > tries * ratio_skips



