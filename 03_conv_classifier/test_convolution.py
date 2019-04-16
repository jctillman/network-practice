
from numpy.random import random, randint, rand

import numpy as np
from convolution import ConvolutionMult, Convolution, MaxPool
from loss import mean_squared_loss


def generic_test_Convolution_dimensions(conv_cls):

    for _ in range(10):
        batch_size = randint(1, 4)
        width = randint(5, 10)
        height = randint(5, 10)
        input_c = randint(1, 3)
        rand_input = rand(batch_size, width, height, input_c)

        kernel = 3
        output_c = randint(1, 5)
        conv = conv_cls((width, height, input_c), kernel, output_c)

        result = conv.forward(rand_input)

        assert result.shape[0] == batch_size
        assert result.shape[1] == width - int(kernel - 1)
        assert result.shape[2] == height - int(kernel - 1)
        assert result.shape[3] == output_c

def module_test_matrices(times):

    for _ in range(times):
        batch_size = randint(low=1, high=3)
        
        width = randint(low=5, high=13)
        height = randint(low=5, high=13)
        input_c = randint(low=1, high=3)
        
        output_c = randint(1, 4)

        inp = rand(batch_size, width, height, input_c) * 0.5 + 0.25

        out = rand(batch_size, width - 2, height - 2, output_c) * 0.5 + 0.25
        
        yield batch_size, width, height, input_c, output_c, inp, out


def generic_test_Convolution_optimize(conv_cls):
    
    LR = 0.0001
    tries = 200
    failures = 0
    for bs, w, h, inp_c, out_c, inp, truth in module_test_matrices(tries):
       
        model = conv_cls((w, h, inp_c), 3, out_c)
        outputs = model.forward(inp)
        old_loss, derivative = mean_squared_loss(prediction=outputs, truth=truth)
        model.backward(derivative)
        model.optimize(LR)

        new_outputs = model.forward(inp)
        new_loss, derivative = mean_squared_loss(prediction=new_outputs, truth=truth)
        if new_loss >= old_loss:
            failures = failures + 1
    
    assert failures <= tries / 5


def generic_test_Convolution_derivative_amount(conv_cls):
    
    LR = 0.0000001
    tries = 100
    failures = 0
    skips = 0
    for bs, w, h, inp_c, out_c, inputs, truth in module_test_matrices(tries):
       
        model = conv_cls((w, h, inp_c), 3, out_c)
        outputs = model.forward(inputs)
        old_loss, derivative = mean_squared_loss(prediction=outputs, truth=truth)
        input_deriv = model.backward(derivative)
        
        indiv_w = randint(0, w)
        indiv_h = randint(0, h)
        spot = randint(0, inp_c)

        # Algo breaks down where derivative is mostly flat
        in_d = input_deriv
        if np.any(np.abs(in_d[:,indiv_w,indiv_h,spot]) < 0.002):
            skips += 1
            continue

        for bs_l in range(bs):
            change_amount = (1 / input_deriv[bs_l][indiv_w][indiv_h][spot]) * LR
            inputs[bs_l][indiv_w][indiv_h][spot] = (
                inputs[bs_l][indiv_w][indiv_h][spot] - change_amount)


        new_outputs = model.forward(inputs)
        new_loss, derivative = mean_squared_loss(prediction=new_outputs, truth=truth)
        
        assert np.isclose(
            (old_loss - new_loss) * (1 / LR),
            1 * bs,
            atol=0.02) or (old_loss == 0.0 and new_loss == 0.0)
    
    assert skips < tries / 3


def generic_test_Convolution_derivative_amount(conv_cls):
    
    LR = 0.0000001
    tries = 100
    failures = 0
    skips = 0
    for bs, w, h, inp_c, out_c, inputs, truth in module_test_matrices(tries):
       
        model = conv_cls((w, h, inp_c), 3, out_c)
        outputs = model.forward(inputs)
        old_loss, derivative = mean_squared_loss(prediction=outputs, truth=truth)
        input_deriv = model.backward(derivative)
        
        indiv_w = randint(0, w)
        indiv_h = randint(0, h)
        spot = randint(0, inp_c)

        # Algo breaks down where derivative is mostly flat
        in_d = input_deriv
        if np.any(np.abs(in_d[:,indiv_w,indiv_h,spot]) < 0.002):
            skips += 1
            continue

        for bs_l in range(bs):
            change_amount = (1 / input_deriv[bs_l][indiv_w][indiv_h][spot]) * LR
            inputs[bs_l][indiv_w][indiv_h][spot] = (
                inputs[bs_l][indiv_w][indiv_h][spot] - change_amount)


        new_outputs = model.forward(inputs)
        new_loss, derivative = mean_squared_loss(prediction=new_outputs, truth=truth)
        
        assert np.isclose(
            (old_loss - new_loss) * (1 / LR),
            1 * bs,
            atol=0.02) or (old_loss == 0.0 and new_loss == 0.0)
    
    assert skips < tries / 3



def test_Convolution_vanilla():
    generic_test_Convolution_dimensions(ConvolutionMult)
    generic_test_Convolution_optimize(ConvolutionMult)
    generic_test_Convolution_derivative_amount(ConvolutionMult)

def test_Convolution():
    generic_test_Convolution_dimensions(Convolution)
    generic_test_Convolution_optimize(Convolution)
    generic_test_Convolution_derivative_amount(Convolution)


def module_test_matrices_pool(times):

    for _ in range(times):
        batch_size = randint(low=1, high=3)
        
        width = 12
        height = 12
        input_c = randint(low=1, high=3)
        
        size = randint(low=1, high=4)

        inp = rand(batch_size, width, height, input_c) * 0.5 + 0.25

        out = rand(batch_size, int(width / size), int(height/ size), input_c) * 0.5 + 0.25
        
        yield batch_size, width, height, input_c, inp, out, size


def generic_test_Pool_derivative_amount(pool):
    
    LR = 0.0000001
    tries = 100
    failures = 0
    skips = 0
    for bs, w, h, input_c, inp, truth, size in module_test_matrices_pool(tries):
       
        model = pool(size)
        outputs = model.forward(inp)
        old_loss, derivative = mean_squared_loss(prediction=outputs, truth=truth)
        input_deriv = model.backward(derivative)
        
        indiv_w = randint(0, w / size)
        indiv_h = randint(0, h / size)
        spot = randint(0, input_c)

        # Algo breaks down where derivative is mostly flat
        in_d = input_deriv
        if np.any(np.abs(in_d[:,indiv_w,indiv_h,spot]) < 0.002):
            skips += 1
            continue

        for bs_l in range(bs):
            change_amount = (1 / input_deriv[bs_l][indiv_w][indiv_h][spot]) * LR
            inp[bs_l][indiv_w][indiv_h][spot] = (
                inp[bs_l][indiv_w][indiv_h][spot] - change_amount)


        new_outputs = model.forward(inp)
        new_loss, derivative = mean_squared_loss(prediction=new_outputs, truth=truth)
        
        assert np.isclose(
            (old_loss - new_loss) * (1 / LR),
            1 * bs,
            atol=0.02) or (old_loss == 0.0 and new_loss == 0.0)
    
    assert skips < tries / 1.5

def test_MaxPool():

    # Super basic testing
    inp = np.array([[[[1],[2]],[[3],[4]]]])
    m = MaxPool(2)
    res = m.forward(inp)
    assert res.reshape(-1) == 4
    m = MaxPool(1)
    assert np.array_equal(m.forward(inp), inp)

    generic_test_Pool_derivative_amount(MaxPool)


