def sgd_optimizer(weights, gradients):
    for key in set(set(weights.keys()) & set(gradients.keys())):
        weights[key] = weights[key] - (gradients[key] * 0.001)