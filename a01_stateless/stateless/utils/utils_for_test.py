import numpy as np

from stateless.loss.loss import mean_squared_loss
from math import floor


def dict_assign(arr, shape_dict):
    
    for i in range(len(arr)):
        tokens = arr[i].split(r'\+\*')
        for token in tokens:
            if token not in shape_dict:
                shape_dict[token] = np.random.randint(low=1, high=10)

# TODO:
# Modify so that instead of just getting arrays like
# ['N','M','O'] it can also handle ['N*M','O']
# and things like that.
def shape_from_format(inputs, outputs):

    def return_func():
        shape_dict = {}

        for i in range(len(inputs)):
            dict_assign(inputs[i], shape_dict)
        dict_assign(outputs, shape_dict)

        def get_val(string):
            value = shape_dict[string[0]]
            prev_op = None
            for char in string[1:]:
                if char == '+':
                    prev_op = '+'
                elif char == '*':
                    prev_op = '*'
                elif char in shape_dict:
                    if prev_op == '*':
                        value = value * shape_dict[char]
                    else:
                        value = value + shape_dict[char]
                else:
                    raise Error("!")
            return value

        input_shape = [ [ get_val(y) for y in x] for x in inputs ]
        output_shape = [ get_val(x) for x in outputs ]

        print(input_shape, output_shape)
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

def do_test_derivs(
        tested,
        shape_generator,
        LR = 0.00001,
        tries = 100,
        permissible_skip_ratio=0.5):

    skip = 0
    for inputs, outputs in gen_test_data(100, shape_generator):
        
        results = tested.forward(*inputs)
        old_loss, error = mean_squared_loss(prediction=results, truth=outputs)
        errors = tested.backward(inputs=inputs, outputs=results, error=error) 

        copied_inputs = [ x.copy() for x in inputs ]

        for input_index, (inp, err) in enumerate(zip(inputs, errors)):
        
            random_spot = [ int(floor(x * np.random.random())) for x in inp.shape ]
            
            this_err = err.copy()
            for i in range(len(random_spot)):
                this_err = this_err[random_spot[i]]

            if np.abs(this_err) < 0.05:
                skip = skip + 1
                break
            
            indiv = random_spot
            change_amount = (1 / this_err) * LR
            if len(indiv) == 1:
                inp[indiv[0]] = inp[indiv[0]] - change_amount
            if len(indiv) == 2:
                inp[indiv[0]][indiv[1]] = inp[indiv[0]][indiv[1]] - change_amount
            if len(indiv) == 3:
                inp[indiv[0]][indiv[1]][indiv[2]] = inp[indiv[0]][indiv[1]][indiv[2]] - change_amount
            if len(indiv) == 4:
                inp[indiv[0]][indiv[1]][indiv[2]][indiv[3]] = inp[indiv[0]][indiv[1]][indiv[2]][indiv[3]] - change_amount

            new_outputs = tested.forward(*[
                x if i != input_index else inp for i, x in enumerate(copied_inputs)
            ])

            new_loss, derivative = mean_squared_loss(prediction=new_outputs, truth=outputs)
            
            assert np.isclose(
                (old_loss - new_loss) * (1 / LR),
                1,
                atol=0.01) or (old_loss == 0.0 and new_loss == 0.0)
    
    assert skip < tries * permissible_skip_ratio



