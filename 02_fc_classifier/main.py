
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

import tensorflow as tf
mnist = tf.keras.datasets.mnist

def mnist_generator(batch_size, train=True):

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    xs, ys = None, None
    if train == True:
        xs = x_train / 255 - 0.5
        ys = y_train / 255 - 0.5
    else:
        xs = x_test / 255 - 0.5
        ys = y_train / 255 - 0.5

    while True:
        b_xs = []
        b_ys = []
        for _ in range(batch_size):
            b_xs.append(xs[np.random.randint(0, len(xs))])
            b_ys.append(ys[np.random.randint(0, len(ys))])
        b_xs = np.array(b_xs) 
        b_ys = np.array(b_ys)
        yield b_xs, b_ys

    assert False


def tree_height_generator(batch_size):
    '''
    Going to return data_x, data_y
    data_x is of [bs, 3] dimensions
    data_y is of [bs, 1] dimensions
    '''
    def get_tuples():
        x_grow_time = np.random.random() * 3
        x_is_transplant = 1 if np.random.random() > 0.5 else 0
        x_is_spruce = 1 if np.random.random() > 0.5 else 0
        grow_factor = 1 if x_is_spruce == 1 else 3
        inp = [x_grow_time, x_is_transplant, x_is_spruce]
        out = [x_grow_time * grow_factor + x_is_transplant + 0.5]
        return (inp, out)

    while True:
        inp, out = zip(*[ get_tuples() for _ in range(batch_size) ])
        yield np.array(inp), np.array(out)

def tree_kind_generator(batch_size):
    '''
    Going to return data_x, data_y
    data_x is of [bs, 3] dimensions
    data_y is of [bs, 1] dimensions
    '''
    def get_tuples():
        
        x_is_spruce = 1 if np.random.random() > 0.5 else 0
        x_grow_time = np.random.random()

        x_height, x_greenness = None, None
        if x_is_spruce:
            x_height = x_grow_time * 0.5
            x_greenness = 0.3 * x_grow_time
        else:
            x_height = x_grow_time * 0.3
            x_greenness = 0.6 * x_grow_time
            
        inp = [x_grow_time, x_height, x_greenness]
        out = [x_is_spruce, 1 - x_is_spruce]
        return (inp, out)

    while True:
        inp, out = zip(*[ get_tuples() for _ in range(batch_size) ])
        yield np.array(inp), np.array(out)


def mean_squared_loss(prediction=None, truth=None):
    assert prediction.shape == truth.shape
    loss = 0.5 * np.sum(np.power(prediction - truth, 2))
    deriv = prediction - truth
    return loss, deriv

class DefaultLogger:

    def __init__(self, file_name):
        self.file_name = file_name
        self.info = {}


    def __call__(self, values):
        for name, value in values: 
            if name not in self.info:
                self.info[name] = []
            self.info[name].append(value)

    def close(self, main_key):
        csv_name = self.file_name + '.csv'
        open(csv_name, 'w').close()

        file = open(csv_name, 'a')
        writer = csv.writer(file, delimiter=',')
        
        keys = list(self.info.keys())
        writer.writerow(keys)
        for i in range(len(self.info[keys[0]])):
            writer.writerow([ self.info[k][i] for k in keys])


        df = pd.read_csv(csv_name)
        less_main = list(set(keys) - set([main_key]))
        for other_key in less_main:
            plt.plot(df[main_key], df[other_key])

        plt.savefig(self.file_name+'.png')

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


class FullyConnectedModel(object):

    def __init__(self, shape):

        hidden_size = 12
        self.shape = shape
        input_size, output_size = shape
        self.input_size = input_size
        self.output_size = output_size
        self.modules = [
            MatrixMult([input_size, hidden_size]),
            MatrixAdd([hidden_size]),
            Relu(),
            MatrixMult([hidden_size, hidden_size]),
            MatrixAdd([hidden_size]),
            Relu(),
            MatrixMult([hidden_size, output_size]),
            MatrixAdd([output_size]),
            Softmax(),
        ]

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

def step_based_training_loop(
    model=None,
    loss=None,
    train_gen=None,
    test_gen=None,
    total_steps=5000,
    test_frequency=10,
    learning_rate=0.01,
    logger=DefaultLogger('untitled_run')):

    for n in range(total_steps):

        data_x, data_y = next(train_gen)
        pred = model.forward(data_x)
        train_loss, deriv = loss(pred, data_y)
        model.backward(deriv)
        model.optimize(learning_rate)
        if n % test_frequency == 0:
            test_x, test_y = next(test_gen)
            test_pred = model.forward(test_x)
            print(test_pred)
            test_loss, _ = loss(test_pred, test_y)
            logger([
                ('step', n),
                ('train_loss', train_loss),
                ('test_loss', test_loss)])

    logger.close('step')
    return model
        

def main():
    batch_size = 10
    input_size = 3
    hidden_size = 15
    output_size = 2
    trained_model = step_based_training_loop(
        model=FullyConnectedModel([input_size, output_size]),
        loss=mean_squared_loss,
        train_gen=tree_kind_generator(batch_size),
        test_gen=tree_kind_generator(batch_size),
    )

if __name__ == '__main__': 
    main()
