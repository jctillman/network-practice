import uuid
import numpy as np

from dag import Dag

class LinkedAbstract:
    
    def __init__(self):
        pass

def data_equality(n1, n2):
    return (
        n1['sgc'] == n2['sgc'] and
        len(n1['input_names']) == len(n2['input_names']) and 
        all([
            x == y for x, y
            in zip(n1['input_names'], n2['input_names'])
        ])
    )

def sum_arr(arr):
    ret = np.zeros(arr[0].shape)
    for i in arr:
        ret = ret + i
    return ret

def linker(sgc_cls):
    
    class LinkedStateless(LinkedAbstract):

        def __init__(self, inputs, name = None):

            # Guarantee that
            # 1. `inputs` is list of subclasss LinkedAbstract
            # 2. `name` is unique string.
            name = name or str(uuid.uuid4())
            inputs = inputs if isinstance(inputs, list) else [inputs]
            for inp in inputs:
                assert issubclass(type(inp), LinkedAbstract)

            self.name = name
            
            # `data` defines each node of our dag, and
            # contains
            # 1. `sgc_cls`, which is forward / backward
            # 2. `input_names` ordered list of parent names.
            input_names = [ inp.name for inp in inputs ]
            data = {
                'sgc': sgc_cls,
                'input_names': input_names
            }

            # Create the dag which actually works
            self.dag = Dag()
            self.dag.add_node(name, input_names, data)
            for dag in [ x.dag for x in inputs ]:
                self.dag.merge_dag(dag, data_equality)

        def get_names(self):
            return self.dag.get_node_names()

        def get_inputs(self):
            return self.dag.get_nothing_upstream()

        def get_outputs(self):
            return self.dag.get_nothing_downstream()
        
        def get_inputs_required_for(self, names_lst):
            all_required_for = self.dag.get_upstreams(names_lst)
            potential_inputs = self.get_inputs()
            return [ x for x in potential_inputs if x in all_required_for ]

        def _verify_input_dict(self, input_keys, output_keys):
            required_for = self.get_inputs_required_for(output_keys)
            return all([ x in input_keys for x in required_for ])
        
        '''
            Input:
                input_dict { 'name': np.arr, ... }
                outputs: [ 'name4', 'name5' ]
            Outputs:
                { 'name4': np.arr, 'name5': np.arr }
        '''
        def forw(self, input_dict, outputs=[]):

            # What do we want to return NP arrays for?
            outputs = self.get_outputs() if len(outputs) == 0 else outputs
            assert self._verify_input_dict(input_dict.keys(), outputs)
            
            # Get list of keys that need to be calculated,
            # in order that they need to be calculated.
            input_keys = input_dict.keys()
            must_get = self.dag.get_upstreams(outputs)
            ordered = [
                x for x in self.dag.ordered_from_top() 
                if (
                    x in must_get and
                    x not in input_keys
                ) 
            ]

            output_dict = input_dict.copy()
            for key in ordered:
                data = self.dag.get_node(key).data
                fnc = data['sgc'].forward
                input_names= data['input_names']
                output_dict[key] = fnc(*[
                    output_dict[k] for k in input_names
                ])
                        
            return output_dict
        
        def find_to_calc_back(self, derivative_keys, output_derivs):
            # Todo: make this sooo much prettier
            upstream_of_derivs = self.dag.get_upstreams(derivative_keys)
            downstream_of_desired_output = self.dag.get_downstreams(output_derivs)
            to_calc = set(upstream_of_derivs) & set(downstream_of_desired_output)

            for output in output_derivs:
                if not output in upstream_of_derivs:
                    print("Problem ", output, " not in ", upstream_of_derivs)
                    assert False

            return to_calc

        '''
            Input:
               derivative_dict: { 'name3': np.arr, ... }
               current_values: { 'name1': nparr, .. }
               outputs: [ 'name1', 'name2' ]
            Output: { 'name1': np.arr }
        '''
        def back(self,
                derivative_dict,
                current_values,
                output_derivs):
            
            to_calc = self.find_to_calc_back(
                list(derivative_dict.keys()),
                output_derivs)
            # Building_deriv is where
            # most of these are calculated
            build = { key: {} for key in to_calc }
            for key in derivative_dict.keys():
                for parent in self.dag.get_parents(key):
                    build[parent][key] = derivative_dict[key]

            # Create a list of the ordered 
            # elements that we need to calculate
            to_do = to_calc - set(derivative_dict.keys())
            ordered = [
                x for x in self.dag.ordered_from_bottom() 
                if x in to_do
            ]
            
            for key in ordered:

                data = self.dag.get_node(key).data
                
                inp = [ current_values[i] for i in data['input_names'] ]
                if len(data['input_names']) == 0:
                    inp = [ current_values[key] ]

                errors = fnc = data['sgc'].backward(
                    inputs=inp,
                    outputs=current_values[key],
                    error=sum_arr([ build[key][x]
                        for x in self.dag.get_children(key) ]))

                for i, k in enumerate(data['input_names']):
                    if k not in build:
                        build[k] = {}
                    build[k][key] = errors[i]

            ret = {}
            for n in output_derivs:
                keys = list(build[n].keys())
                arr = [ build[n][k] for k in keys ]
                if len(arr) > 0:
                    ret[n] = sum_arr(arr)

            return ret
               
    return LinkedStateless


