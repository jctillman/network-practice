def sgd_optimizer(weights, gradients):
    for key in set(set(weights.keys()) & set(gradients.keys())):
        weights[key] = weights[key] - (gradients[key] * 0.001)

def get_sgd_optimizer(num):

        def sgd_optimizer(weights, gradients):
                for key in set(set(weights.keys()) & set(gradients.keys())):
                        weights[key] = weights[key] - (gradients[key] * num)
        
        return sgd_optimizer
                
        