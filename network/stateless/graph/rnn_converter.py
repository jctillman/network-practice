import numpy as np


# Goals
# 0. Make single step-back
# function, write test verifying that
# it actually works.
# 1. Make function invoking single
# step-back thing repeatedly, which uses
# the outputs from the multiple
# step-back thing.
# 2. 

def is_numpy_dict(dct):
    return (
        isinstance(dct, dict) and
        all([
            isinstance(x, np.ndarray)
            for x in dct.values()
        ])
    )


def forward_rnn_step(
    graph,
    weights,
    initial_hidden_state,
    time_indexed_inputs):
    pis = initial_hidden_state
    assert is_numpy_dict(weights)
    assert is_numpy_dict(initial_hidden_state)
    assert is_numpy_dict(time_indexed_inputs)
    return graph.forw({
       **weights,
       **{'prior_' + k: v for k, v in pis.items()},
       **time_indexed_inputs,
    })

def get_time_slice(inputs, time_step):
    assert is_numpy_dict(inputs)
    return {
        k: v[:,time_step,:] for k, v in inputs.items()
    }

def get_timesteps(inputs):
    values = list(inputs.values())
    assert all([
        val.shape[1] == values[0].shape[1]
        for val in values
    ])
    return values[0].shape[1]

def rnn_forward(
    graph,
    weights,
    initial_hidden_state,
    inputs):
    assert is_numpy_dict(weights)
    assert is_numpy_dict(initial_hidden_state)
    assert is_numpy_dict(inputs)

    values = []
    timesteps = get_timesteps(inputs)
    prior_state = initial_hidden_state
    for i in range(timesteps):
        time_slice = forward_rnn_step(
            graph,
            weights,
            prior_state,
            get_time_slice(inputs, i)
        )
        values.append(time_slice)
        prior_state = time_slice
    return values


def terminal_values_only(
    linked,
    numpy_dict):
    ends = linked.dag.get_without_descendants()
    return all([
        key in ends
        for key in numpy_dict.keys()
    ])

def backward_rnn_step(
    linked,
    time_sliced_value,
    time_sliced_deriv,
    prior_time_sliced,
    output_deriv_keys):
    assert is_numpy_dict(time_sliced_value)
    assert is_numpy_dict(time_sliced_deriv)
    assert is_numpy_dict(prior_time_sliced)
    assert isinstance(output_deriv_keys, list)

    prior_time_sliced_trans = {
        k.replace('prior_',''): v
        for k, v in prior_time_sliced.items()
        if 'prior_' in k
    }
    #print(type(prior_time_sliced_trans))
    #print('y', list(prior_time_sliced_trans.keys()), linked.dag.get_without_descendants())
    assert terminal_values_only(linked, prior_time_sliced_trans)
    assert terminal_values_only(linked, time_sliced_deriv)
    assert not (
        set(prior_time_sliced_trans.keys()) &
        set(time_sliced_deriv.keys())
    )

    combined_derivs = {
        **time_sliced_deriv,
        **prior_time_sliced_trans,
    }

    return linked.back(
        combined_derivs,
        time_sliced_value,
        output_deriv_keys
    )



import numpy as np

def to_rnn(linked):

    class LinkedRNN():

        def __init__(self):
            pass
        
        # Assume all the inputs are like
        # [1, X]
        def forw_create(self, time_steps, input, weights, initial_hidden, adapter ):
            values = [input]
            prior_state = initial_hidden
            for i in range(time_steps):
                time_slice = forward_rnn_step(
                    linked,
                    weights,
                    prior_state,
                    input,
                )
                value = adapter(time_slice)
                input = {
                    k: v for k, v in value.items()
                    if k in input.keys()
                }
                prior_state = value
                values.append(value)
                
            return values

        # Assume all the inputs are like
        # [BATCH, TIME, X]
        def forw(self, inputs, weights, hidden_start):
            return rnn_forward(
                linked,
                weights,
                hidden_start,
                inputs
            )
            
        def back(self,
            time_sliced_values,
            time_sliced_output_derivs,
            output_deriv_keys):

            rev_TSV = list(reversed(time_sliced_values))
            rev_SOD = list(reversed(time_sliced_output_derivs))

            timeSteps = len(time_sliced_values)
            derivs = []
            for i in range(timeSteps):
                
                #all_nodes = linked.get_names()
                #derivative_dict = None
                
                #if i == 0:
                #    derivative_dict = rev_SOD[i]
                #else:
                #    derivative_dict = {
                #        **rev_SOD[i],
                #        **prior_derivs[i - 1]
                #    }

                #latest_deriv = linked.back(
                #    derivative_dict,
                #    rev_TSV[i],
                #    output_deriv_keys,
                #)

                # ("BACK", list(output_deriv_keys))
                latest_deriv = backward_rnn_step(
                    linked,
                    rev_TSV[i],
                    rev_SOD[i],
                    derivs[i - 1] if i else {},
                    list(output_deriv_keys)
                )
                                
                derivs.append(latest_deriv)

            start = derivs[0]
            for i in range(1, len(derivs)):
                for key, value in derivs[i].items():
                    start[key] = start[key] + value

            return start 



    return LinkedRNN()