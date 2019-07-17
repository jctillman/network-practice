import uuid
import numpy as np

from stateless.utils.dag import Dag

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
            # 1. `name` is unique string.
            # 2. `inputs` is list of subclasss LinkedAbstract
            name = name or str(uuid.uuid4())
            assert isinstance(name, str)
            inputs = inputs if isinstance(inputs, list) else [inputs]
            for inp in inputs:
                assert issubclass(type(inp), LinkedAbstract)

            self.name = name
            
            # `data` defines each node of our dag, and
            # contains
            # 1. `sgc_cls`, which is forward / backward
            # 2. `input_names` ORDERED list of parent names.
            input_names = [ inp.name for inp in inputs ]
            data = {
                'sgc': sgc_cls,
                'input_names': input_names
            }

            # Create the dag which actually works
            self.dag = Dag()
            self.dag.add_node(name, set(input_names), data)

            for dag in [ x.dag for x in inputs ]:
                self.dag.merge_dag(dag, data_equality)

            for i in inputs:
                i.dag = self.dag


        def get_names(self):
            return self.dag.get_node_names()

        def get_inputs(self):
            return self.dag.get_without_parents()

        def get_outputs(self):
            return self.dag.get_without_descendants()
        
        def get_inputs_required_for(self, names_lst):
            all_required_for = self.dag.get_ancestors_for_all(names_lst)
            potential_inputs = self.get_inputs()
            return set([ x for x in potential_inputs if x in all_required_for ])

        def _verify_input_dict(self, input_keys, output_keys):
            required_for = self.get_inputs_required_for(output_keys)
            
            valid = all([ x in input_keys for x in required_for ])
            if valid:
                return valid
            else:
                print(output_keys, required_for)
                print([ (x in input_keys, x) for x in required_for ])
        '''
            Input:
                input_dict { 'name': np.arr, ... }
                outputs: [ 'name4', 'name5' ]
            Outputs:
                { 'name4': np.arr, 'name5': np.arr }
        '''
        def forw(self, input_dict, outputs=[]):

            # What do we want to return NP arrays for?
            input_keys = input_dict.keys()
            outputs = self.get_outputs() if len(outputs) == 0 else outputs
            #print('get outputs', self.get_outputs())
            assert self._verify_input_dict(input_keys, outputs)
            for key in outputs:
                assert isinstance(key, str)
            
            # Get list of keys that need to be calculated,
            # in order that they need to be calculated.
            must_get = self.dag.get_ancestors_for_all(outputs)
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
        
        def find_to_calc_back(self, derivative_keys, output_keys):
            # We need all inputs [upstream] of desired derivatives
            # in order to calculate these derivatives.
            upstream_of_derivs = self.dag.get_ancestors_for_all(derivative_keys)
            #print("TO_CALC", output_keys, upstream_of_derivs)
            assert all([key in upstream_of_derivs for key in output_keys ])
            
            #downstream_of_desired_output = self.dag.get_descendants_for_all(output_keys)
            #to_calc = set(upstream_of_derivs) & set(downstream_of_desired_output)
            return set(upstream_of_derivs)

        '''
            Input:
               derivative_dict: { 'name3': np.arr, ... }
               values_dict: { 'name1': np.arr, .. }
               outputs: [ 'name1', 'name2' ]
            Output: { 'name1': np.arr }
        '''
        def back(self,
                derivative_dict,
                values_dict,
                output_derivs):

            to_calc = self.find_to_calc_back(
                list(derivative_dict.keys()),
                output_derivs)
            #print(to_calc, derivative_dict.keys(), output_derivs)

            # "build" is a dict of dicts, with
            # [parent][child] keys and terminal values
            # as the gradients coming from child to parent.
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
            
            # Calculate data, going backward
            # through the graph structure.
            for key in ordered:
                data = self.dag.get_node(key).data
                
                inp = None
                if len(data['input_names']) == 0:
                    inp = [ values_dict[key] ]
                else:
                    inp = [ values_dict[i] for i in data['input_names'] ]
                
                #print('key', key)
                #print(self.dag.get_node_names())
                #print(self.dag.get_children(key))
                errors = data['sgc'].backward(
                    inputs=inp,
                    outputs=values_dict[key],
                    error=sum_arr([ build[key][x]
                        for x in self.dag.get_children(key)
                        if key in build and x in build[key]
                    ]))

                for index, input_key in enumerate(data['input_names']):
                    #if input_key not in build:
                    #    build[input_key] = {}
                    build[input_key][key] = errors[index]

            ret = {}
            for n in output_derivs:
                children = list(build[n].keys())
                arr = [ build[n][child_key] for child_key in children ]
                if len(arr) > 0:
                    ret[n] = sum_arr(arr)

            return ret
               
    return LinkedStateless


