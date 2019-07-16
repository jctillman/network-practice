import numpy as np

from stateless.graph.rnn_converter import (
    is_numpy_dict,
    forward_rnn_step,
    get_time_slice,
    get_timesteps,
    rnn_forward,
)

from stateless.graph.graph_linked import (
    Relu,
    Exponent,
    Identity,
    Parameter,
    Probabilize,
    Input,
    Concat,
    Sigmoid,
    Prior,
    MatrixAdd,
    MatrixMult,
    MatrixAddExact)

def test_is_numpy_dict():

    assert is_numpy_dict({
        'one': np.array([1,2,3])
    })

    assert not is_numpy_dict({
        'one': np.array([1,2,3]),
        'two': None
    })

    assert is_numpy_dict({})

def get_random_rnn():
    weights1 = Parameter(name="weights1")
    inputs = Input(name='input')

    prior_hidden = Prior(name='hidden')
    inputs_concat = Concat([inputs, prior_hidden], name='concat')

    hidden_internal = Relu([
        MatrixMult([inputs, weights1], name='matrixmult')
        ], name='hidden_internal')
    
    hidden = Identity([hidden_internal], name='hidden')

    weights2 = Parameter(name="weights2")
    output = Relu([MatrixMult([hidden_internal, weights2], name='matrixmult2')], name='output')
    print(list(output.dag.get_node_names()))
    return output

def get_random_rnn_values(time_length=None):
    BN = 3
    INPUT_SIZE = 4
    HIDDEN_SIZE = 8
    OUTPUT_SIZE = 5
    rand = np.random.rand
    weights = {
        'weights1': rand(INPUT_SIZE, HIDDEN_SIZE),
        'weights2': rand(HIDDEN_SIZE, OUTPUT_SIZE),
    }
    prior_internal_state = {
        'hidden': rand(BN, HIDDEN_SIZE),
    }
    time_indexed_inputs = {
        'input': rand(BN, INPUT_SIZE)
    }
    if time_length is not None:
        time_indexed_inputs = {
            'input': rand(BN, time_length, INPUT_SIZE)
        }

    return weights, prior_internal_state, time_indexed_inputs

def test_forward_rnn_step():
    output = get_random_rnn()
    weights, prior_internal_state, time_indexed_inputs = get_random_rnn_values()
    results = forward_rnn_step(
        output, weights,
        prior_internal_state,
        time_indexed_inputs
    )
    print(list(results.keys()))
    assert 'output' in results
    assert 'hidden' in results
    assert is_numpy_dict(results)

def test_get_time_slice():
    rand = np.random.rand
    sliced = get_time_slice({
        'inputs': rand(4, 5, 6)
    }, 2)
    assert sliced['inputs'].shape[0] == 4
    assert sliced['inputs'].shape[1] == 6
    assert len(sliced['inputs'].shape) == 2

def test_get_timesteps():
    rand = np.random.rand
    ts1 = get_timesteps({
        'inputs': rand(4, 5, 6)
    })
    ts2 = get_timesteps({
        'inputs': rand(4, 7, 6)
    })
    assert ts1 == 5
    assert ts2 == 7

def test_rnn_forward():

    output = get_random_rnn()
    weights, prior_internal_state, inputs = get_random_rnn_values(10)

    results = rnn_forward(
        output,
        weights,
        prior_internal_state,
        inputs
    )
    assert len(results) == 10
    





