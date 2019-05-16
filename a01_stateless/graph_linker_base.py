import uuid
import numpy as np

from dag import Dag

class LinkedAbstract:
    
    def __init__(self):
        pass


def linker(sgc_cls):
    
    class LinkedStateless(LinkedAbstract):

        def __init__(self, inputs, name = None):

            inputs = inputs if isinstance(inputs, list) else [inputs]
            for inp in inputs:
                assert issubclass(type(inp), LinkedAbstract)

            name = name or str(uuid.uuid4())
            self.name = name
            
            input_names = [ inp.name for inp in inputs ]
            # 'input_names' is guaranteed to be unmodified / preserve
            # order, which dag edges isn't.  Kinda ugly.
            data = { 'sgc': sgc_cls, 'input_names': input_names }


            self.dag = Dag()
            self.dag.add_node(name, input_names, data)
            for dag in [ x.dag for x in inputs ]:
                self.dag.merge_dag(dag)

        def get_names(self):
            return self.dag.get_node_names()

        def get_inputs(self):
            return self.dag.get_nothing_upstream()

        def get_outputs(self):
            return self.dag.get_nothing_downstream()
        
        def get_all_required_for(self, names_lst):
            return self.dag.get_upstreams(names_lst)

        def get_inputs_required_for(self, names_lst):
            all_required_for = self.get_all_required_for(names_lst)
            potential_inputs = self.get_inputs()
            return [ x for x in potential_inputs if x in all_required_for ]

        def _verify_input_dict(self, input_keys, output_keys):
            required_for = self.get_inputs_required_for(output_keys)
            return all([ x in input_keys for x in required_for ])

        def _sum_arr(self, arr):
            ret = np.zeros(arr[0].shape)
            for i in arr:
                ret = ret + i
            return ret
        
        '''
            Input:
                input_dict 
                    {
                        'name': np.arr,
                        'name2': np.arr,
                    }
                outputs:
                    [ 'name4', 'name5' ]
            Outputs:
                {
                    'name4': np.arr,
                    'name5': np.arr,
                }
        '''
        def forw(self, input_dict, outputs=[]):

            # What do we want to return NP arrays for?
            outputs = self.get_outputs() if len(outputs) == 0 else outputs

            assert self._verify_input_dict(input_dict.keys(), outputs)

            must_get = self.get_all_required_for(outputs)
            output_dict = input_dict.copy()
            
            while len(must_get) > 0:

                for key in must_get:

                    data = self.dag.get_node(key).data
                    fnc = data['sgc'].forward
                    input_names= data['input_names']

                    ready = [ i in output_dict for i in input_names ]
                    
                    if all(ready):

                        must_get.remove(key)

                        if key in input_dict:
                            output_dict[key] = fnc(input_dict[key])
                        else:
                            output_dict[key] = fnc(*[
                                output_dict[k] for k in input_names
                            ])
                        
            return output_dict
        
        '''
            Input:
               derivative_dict:
                    {
                        'name3': np.arr,
                        'name2': np.arr,
                    }
                current_values:
                    {
                        'name1': nparr,
                        'name2': nparr,
                    }
                outputs:
                    [ 'name1', 'name2' ]
            Output:
                {
                    'name1': np.arr,
                    'name2': np.arr,
                }
        '''
        def back(self,
                derivative_dict,
                current_values,
                output_derivs):
            
            # Todo: make this sooo much prettier
            upstream_of_derivs = self.dag.get_upstreams(list(derivative_dict.keys()))
            downstream_of_desired_output = self.dag.get_downstreams(output_derivs)
            to_calc = (set(upstream_of_derivs) & set(downstream_of_desired_output))
            
            # There must be some derivative input
            for output in output_derivs:
                assert output in upstream_of_derivs
           
            # Building_deriv is where
            # most of these are calculated
            build = {}
            for key in to_calc:
                build[key] = {}

            for key in derivative_dict.keys():
                parents = self.dag.get_parents(key)
                for parent in parents:
                    build[parent][key] = derivative_dict[key]

            to_do = to_calc - set(derivative_dict.keys())
            
            # Condition of doing some backprop is that 
            # [to_backprop][input_key] for each is
            # filled in the build.
            # Then:
            #  1. All those summed.
            #  2. Backprop
            #  3. [parent_key][this_key] for parents filled

            while len(to_do) > 0:

                for key in to_do.copy():

                    children = self.dag.get_children(key)

                    ready = all([ i in build[key] for i in children ])

                    if ready:
                        
                        to_do.remove(key)

                        node = self.dag.get_node(key)
                        parents = node.data['input_names']
                        fnc = node.data['sgc'].backward
                        
                        inp = [ current_values[i] for i in parents ]
                        if len(parents) == 0:
                            inp = [ current_values[key] ]

                        error = self._sum_arr([ build[key][x] for x in children ])
                        
                        errors = fnc(inputs=inp,
                            outputs=current_values[key],
                            error=error)

                        for i, k in enumerate(parents):
                            if k not in build:
                                build[k] = {}
                            build[k][key] = errors[i]

            ret = {}
            for n in output_derivs:
                keys = list(build[n].keys())
                arr = [ build[n][k] for k in keys ]
                ret[n] = self._sum_arr(arr)

            return ret
               
    return LinkedStateless


