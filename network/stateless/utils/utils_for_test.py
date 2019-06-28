import numpy as np

from stateless.loss.loss import mean_squared_loss
from math import floor


def dict_assign(arr, shape_dict):
    
    for i in range(len(arr)):
        for token in arr[i].split(r'\+\*'):
            if token not in shape_dict:
                shape_dict[token] = np.random.randint(low=1, high=10)

def get_val(string, shape_dict):
    value = shape_dict[string[0]]
    prev_op = None
    for char in string[1:]:
        if char in '*-+':
            prev_op = char
        elif char in shape_dict:
            if prev_op == '*':
                value = value * shape_dict[char]
            elif prev_op == '-':
                value = value - shape_dict[char]
            else:
                value = value + shape_dict[char]
        else:
            raise Error("!")
    return value

def shape_from_format(inputs, outputs):

    def return_func():
        shape_dict = {}
        for i in range(len(inputs)):
            dict_assign(inputs[i], shape_dict)
        dict_assign(outputs, shape_dict)
        input_shape = [ [ get_val(y, shape_dict) for y in x] for x in inputs ]
        output_shape = [ get_val(x, shape_dict) for x in outputs ]
        return input_shape, output_shape

    return return_func


def gen_test_data(times, datagen_shape):

    for i in range(times):
        chosen_format = datagen_shape()
        chosen_inputs = chosen_format[0].copy()
        chosen_outputs = chosen_format[1].copy()
        inputs = [ np.random.rand(*x) for x in chosen_inputs ]
        outputs = np.random.rand(*chosen_outputs)
        yield [ inputs, outputs ]


def do_test_runs(tested, shape_generator):

    for inputs, outputs in gen_test_data(10, shape_generator):
        results = tested.forward(*inputs)

def get_random_spot(inp, err):
    random_spot = [ int(floor(x * np.random.random())) for x in inp.shape ]
    this_err = err.copy()
    for i in range(len(random_spot)):
        this_err = this_err[random_spot[i]]

    return random_spot, this_err

def change_input_by_amount(inp, loc, change_amount):
    if len(loc) == 1:
        inp[loc[0]] = inp[loc[0]] - change_amount
    if len(loc) == 2:
        inp[loc[0]][loc[1]] = inp[loc[0]][loc[1]] - change_amount
    if len(loc) == 3:
        inp[loc[0]][loc[1]][loc[2]] = inp[loc[0]][loc[1]][loc[2]] - change_amount
    if len(loc) == 4:
        inp[loc[0]][loc[1]][loc[2]][loc[3]] = inp[loc[0]][loc[1]][loc[2]][loc[3]] - change_amount
    if len(loc) == 5:
        inp[loc[0]][loc[1]][loc[2]][loc[3]][loc[4]] = inp[loc[0]][loc[1]][loc[2]][loc[3]][loc[4]] - change_amount


def do_test_derivs(
        tested,
        shape_generator,
        LR = 0.00001,
        tries = 10,
        permissible_skip_ratio=0.5):

    skip = 0
    for inputs, outputs in gen_test_data(tries, shape_generator):
        
        results = tested.forward(*inputs)
        old_loss, error = mean_squared_loss(prediction=results, truth=outputs)
        errors = tested.backward(inputs=inputs, outputs=results, error=error) 

        copied_inputs = [ x.copy() for x in inputs ]

        for input_index, (inp, err) in enumerate(zip(inputs, errors)):
        
            random_spot, this_err = get_random_spot(inp, err)

            if np.abs(this_err) < 0.01:
                skip = skip + 1
                break
            
            change_input_by_amount(inp, random_spot, (1 / this_err) * LR)

            new_outputs = tested.forward(*[
                x if i != input_index else inp for i, x in enumerate(copied_inputs)
            ])

            new_loss, derivative = mean_squared_loss(prediction=new_outputs, truth=outputs)
            
            assert np.isclose(
                (old_loss - new_loss) * (1 / LR),
                1,
                rtol=0.05) or (old_loss == 0.0 and new_loss == 0.0)
    
    assert skip < tries * permissible_skip_ratio



