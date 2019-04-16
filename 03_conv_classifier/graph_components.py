import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

class GraphComponent:

    def __init__(self):
        self.params = self._get_params()
        self.derivs = {}
        for k, v in self.params.items():
            self.derivs[k] = np.zeros(v.shape)
        self.variables = list(self.params.keys())

    def optimize(self, learning_rate):
        for k in self.variables:
            assert self.derivs[k].shape == self.params[k].shape
            self.params[k] = self.params[k] - self.derivs[k] * learning_rate

    def forward(self, input):
        raise Error()

    def backward(self, input):
        raise Error()

    def _get_params(self):
        return {}

class Reshape(GraphComponent):

    def __init__(self, og=None, to=None):
        self.og = og
        self.to = to
        super().__init__()

    def forward(self, input):
        w = np.reshape(input, (-1,) + self.to)
        return w

    def backward(self, derivative):
        return np.reshape(derivative, (-1,) + self.og)


class MatrixMult(GraphComponent):

    def __init__(self, shape):

        self.shape = shape
        self.input_size = shape[0]
        self.output_size = shape[1]
        super().__init__()

    def _get_params(self):
        return {
            'W': np.random.rand(self.input_size, self.output_size) - 0.5
        }

    def forward(self, input):
        assert input.shape[1] == self.input_size
        self.input = input
        self.output = np.matmul(input, self.params['W'])
        return self.output

    def backward(self, derivative):
        
        self.derivs['W'] = np.matmul(self.input.T, derivative)
        assert self.derivs['W'].shape == self.params['W'].shape        
        
        self.error = np.matmul(derivative, self.params['W'].T)
        return self.error



class MatrixAdd(GraphComponent):

    def __init__(self, shape):
        self.shape = shape
        super().__init__()

    def _get_params(self):
        return { 'B': np.zeros(self.shape) }

    def forward(self, input):
        self.input = input
        return input + self.params['B']

    def backward(self, derivative):
        self.derivs['B'] = np.sum(derivative, axis=0)
        return derivative



class Sigmoid(GraphComponent):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.input = input
        exp = np.exp(input)
        self.output = np.divide(exp, exp + 1)
        return self.output

    def backward(self, derivative):
        self.derivs['B'] = np.multiply(self.output, 1 - self.output)
        return np.multiply(self.derivs['B'], derivative)
       

class Relu(GraphComponent):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.input = input
        return np.clip(input, a_min=0, a_max=None)

    def backward(self, derivative):
        self.error = np.where(
            self.input > 0.0, derivative, np.zeros(derivative.shape))
        return self.error

class LeakyRelu(GraphComponent):

    def __init__(self, alpha = 0.1):
        self.alpha = alpha
        super().__init__()

    def forward(self, input):
        self.input = input
        return np.where(input > 0.0, input, input * self.alpha)

    def backward(self, derivative):
        self.error = np.where(
            self.input > 0.0, derivative, derivative * self.alpha)
        return self.error

class Elo(GraphComponent):

    def __init__(self, alpha = 0.1):
        self.alpha = alpha
        super().__init__()

    def forward(self, input):
        self.input = input
        return np.where(input > 0, input, np.exp(input) - 1)

    def backward(self, derivative):
        self.error = np.where(
            self.input > 0.0, derivative, np.multiply(derivative, np.exp(self.input)))
        return self.error

class Exponent(GraphComponent):

    def __init__(self, temperature=1):
        self.temperature = temperature
        super().__init__()

    def forward(self, input):
        self.input = input
        self.output = np.exp(input * self.temperature)
        return self.output

    def backward(self, derivative):
        return np.multiply(self.output, derivative)

class Probabilize(GraphComponent):

    def __init__(self, axis=1):
        self.axis = axis
        super().__init__()

    def forward(self, input):
        self.input = input
        assert np.all(input >= 0)
        self.summed = np.sum(input, axis=self.axis)
        expanded_summed = np.expand_dims(self.summed, axis=self.axis)
        self.output = np.divide(input, expanded_summed)
        return self.output

    def backward(self, derivative):
        
        all_examples = []
        for i in range(self.output.shape[0]):
            ith_case = self.input[i]
            jacobian = []
            total = self.summed[i]
            for ii in range(ith_case.shape[0]):
                result = []
                for iii in range(ith_case.shape[0]):
                    if ii == iii:
                        res = (total - ith_case[ii]) / (total ** 2)
                        result.append(res)
                    else:
                        res = -(ith_case[iii]) / (total ** 2)
                        result.append(res)
                jacobian.append(result)

            result = np.multiply(np.array(jacobian), np.expand_dims(derivative[i], 0))

            all_examples.append(np.sum(np.array(result), axis=1))

        return np.array(all_examples)

class Softmax(GraphComponent):

    def __init__(self, axis=1):
        self.modules = [
            Exponent(),
            Probabilize(axis=axis),
        ]
        super().__init__()

    def forward(self, input):
        self.input = input
        x = input
        for mod in self.modules:
            x = mod.forward(x)
        return x


    def backward(self, derivative):
        error = derivative
        for mod in reversed(self.modules):
            error = mod.backward(error)
        return error


class SequentialModel:
    
    def forward(self, input):
        self.input = input
        x = input
        for mod in self.modules:
            x = mod.forward(x)
        return x


    def backward(self, derivative):
        error = derivative
        for mod in reversed(self.modules):
            error = mod.backward(error)
        return error


    def optimize(self, learning_rate):
        for mod in self.modules:
            mod.optimize(learning_rate)
