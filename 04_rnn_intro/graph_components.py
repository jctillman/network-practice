import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

class GraphComponent:

    def __init__(self):
        self.T = 0
        self.params = self._get_params()
        self.derivs = {}
        for k, v in self.params.items():
            self.derivs[k] = np.zeros(v.shape)
        self.variables = list(self.params.keys())
    
        self.inputs = {}
        self.outputs = {}


    def optimize(self, learning_rate):
        for k in self.variables:
            assert self.derivs[k].shape == self.params[k].shape
            self.params[k] = self.params[k] - self.derivs[k] * learning_rate

    def forward(self, input, T = None):
        
        # Keep track of the time as we move forward
        # and throw if we don't get consecutive time
        # segments
        assert T == None or T == self.T + 1 
        self.T = T or (self.T + 1)
        T = self.T

        # Record input
        self.inputs[T] = input

        # Get results, store them at this time, 
        # and return
        result = self._forward(input, **self.params)
        self.outputs[T] = result
        return result

    def backward(self, derivative):

        error, param_derivs = self._backward(
            derivative,
            **self.params,
            **{"input": self.inputs[self.T]},
            **{"output": self.outputs[self.T]})

        for var in self.variables:
            assert var in param_derivs

        self.derivs = param_derivs
        return error

    def _get_params(self):
        return {}

class Reshape(GraphComponent):

    def __init__(self, og=None, to=None):
        self.og = og
        self.to = to
        super().__init__()

    def _forward(self, input):
        return np.reshape(input, (-1,) + self.to)

    def _backward(self, derivative, input=None, output=None, W=None):
        return np.reshape(derivative, (-1,) + self.og), {}


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

    def _forward(self, input, W=None):
        assert input.shape[1] == self.input_size
        assert W is not None
        return np.matmul(input, W)

    def _backward(self, derivative, input=None, output=None, W=None):
        derivs = { 'W': np.matmul(input.T, derivative) }
        error = np.matmul(derivative, W.T)
        return error, derivs



class MatrixAdd(GraphComponent):

    def __init__(self, shape):
        self.shape = shape
        super().__init__()

    def _get_params(self):
        return { 'B': np.zeros(self.shape) }

    def _forward(self, input, B = None):
        assert B is not None
        return input + B

    def _backward(self, derivative, input=None, output=None, B=None):
        derivs = { 'B': np.sum(derivative, axis=0) }
        error = derivative
        return error, derivs



class Sigmoid(GraphComponent):

    def __init__(self):
        super().__init__()

    def _forward(self, input):
        exp = np.exp(input)
        return np.divide(exp, exp + 1)

    def _backward(self, derivative, input=None, output=None):
        tmp = np.multiply(output, 1 - output)
        return np.multiply(tmp, derivative), {}
       

class Relu(GraphComponent):

    def __init__(self):
        super().__init__()

    def _forward(self, input):
        return np.clip(input, a_min=0, a_max=None)

    def _backward(self, derivative, input=None, output=None):
        error = np.where(input > 0.0, derivative, np.zeros(derivative.shape))
        return error, {}

class LeakyRelu(GraphComponent):

    def __init__(self, alpha = 0.1):
        self.alpha = alpha
        super().__init__()

    def _forward(self, input):
        self.input = input
        return np.where(input > 0.0, input, input * self.alpha)

    def _backward(self, derivative, input=None, output=None):
        error = np.where(input > 0.0, derivative, derivative * self.alpha)
        return error, {}

class Elo(GraphComponent):

    def __init__(self, alpha = 0.1):
        self.alpha = alpha
        super().__init__()

    def _forward(self, input):
        return np.where(input > 0, input, np.exp(input) - 1)

    def _backward(self, derivative, input=None, output=None):
        error = np.where(input > 0.0, derivative, np.multiply(derivative, np.exp(input)))
        return error, {}

class Exponent(GraphComponent):

    def __init__(self, temperature=1):
        self.temperature = temperature
        super().__init__()

    def _forward(self, input):
        return np.exp(input * self.temperature)

    def _backward(self, derivative, input=None, output=None):
        return np.multiply(output, derivative), {}

class Probabilize(GraphComponent):

    def __init__(self, axis=1):
        self.axis = axis
        super().__init__()

    def _forward(self, input):
        assert np.all(input >= 0)
        self.summed = np.sum(input, axis=self.axis)
        expanded_summed = np.expand_dims(self.summed, axis=self.axis)
        return np.divide(input, expanded_summed)

    def _backward(self, derivative, input=None, output=None):
        
        all_examples = []
        for i in range(output.shape[0]):
            ith_case = input[i]
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

        return np.array(all_examples), {}

class SequentialModel:
    
    def forward(self, input):
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

class Softmax(SequentialModel):

    def __init__(self, axis=1):
        self.modules = [
            Exponent(),
            Probabilize(axis=axis),
        ]
        super().__init__()
