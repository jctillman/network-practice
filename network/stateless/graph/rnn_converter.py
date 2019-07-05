import numpy as np

def to_rnn(linked):

    class LinkedRNN():

        def __init__(self):
            pass

        def get_num_timesteps(self, inputs):
            values = list(inputs.values())
            assert all([
                val.shape[1] == values[0].shape[1]
                for val in values
            ])
            return values[0].shape[1]
        
        # Assume all the inputs are like
        # [1, X]
        def forw_create(self, time_steps, input, weights, initial_hidden, adapter ):
            
            ret = [input]
            for ts in range(time_steps):

                inputs_t = {}
                for key, value in input.items():
                    inputs_t[key] = ret[ts][key]

                for key, value in weights.items():
                    inputs_t[key] = value

                for name in linked.get_names():
                    without = name.replace('prior_','')
                    if 'prior_' in name:
                        if ts == 0:
                            
                            inputs_t[name] = initial_hidden[without]
                        else:
                            inputs_t[name] = ret[ts][without]
                
                n = linked.forw(inputs_t)
                ret.append(adapter(n))
            return ret
        # Assume all the inputs are like
        # [BATCH, TIME, X]
        def forw(self, inputs, weights, hidden_start):
            time_steps = self.get_num_timesteps(inputs)
            ret = []
            for ts in range(time_steps):

                inputs_t = {}
                for key, value in inputs.items():
                    inputs_t[key] = value[:,ts,:]

                for key, value in weights.items():
                    inputs_t[key] = value

                for name in linked.get_names():
                    without = name.replace('prior_','')
                    if 'prior_' in name:
                        if ts == 0:
                            inputs_t[name] = hidden_start[without]
                        else:
                            inputs_t[name] = ret[ts - 1][without]
                
                ret.append(linked.forw(inputs_t))
            return ret
            
        def back(self,
            time_sliced_values,
            time_sliced_output_derivs,
            output_derivs):

            rev_TSV = list(reversed(time_sliced_values))
            rev_SOD = list(reversed(time_sliced_output_derivs))

            timeSteps = len(time_sliced_values)
            derivs = []
            prior_derivs = []
            for i in range(timeSteps):
                
                #all_nodes = linked.get_names()
                derivative_dict = None
                
                if i == 0:
                    derivative_dict = rev_SOD[i]
                else:
                    derivative_dict = {
                        **rev_SOD[i],
                        **prior_derivs[i - 1]
                    }

                latest_deriv = linked.back(
                    derivative_dict,
                    rev_TSV[i],
                    output_derivs,
                )
                
                derivs.append(latest_deriv)

                prior_deriv = {}
                for key, value in latest_deriv.items():
                    if 'prior_' in key:
                        prior_deriv[
                            key.replace('prior_', '')
                        ] = value
                
                prior_derivs.append(prior_deriv)

            start = derivs[0]
            for i in range(1, len(derivs)):
                for key, value in derivs[i].items():
                    start[key] = start[key] + value

            return start 



    return LinkedRNN()