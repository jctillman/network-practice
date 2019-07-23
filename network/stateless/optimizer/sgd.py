import numpy as np

def get_sgd_optimizer(num):

        def sgd_optimizer(weights, gradients):
                for key in set(set(weights.keys()) & set(gradients.keys())):
                        weights[key] = np.clip(
                                weights[key] - (gradients[key] * num),
                                a_min=-15,
                                a_max=15)
        
        return sgd_optimizer
                
        