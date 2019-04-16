
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

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


class Relu(GraphComponent):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.input = input
        return np.clip(input, a_min=0, a_max=None)

    def backward(self, derivative):
        self.error = np.where(
            self.input > 0, derivative, np.zeros(derivative.shape))
        return self.error


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
    total_steps=2000,
    test_frequency=10,
    learning_rate=0.0001,
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
            test_loss, _ = loss(test_pred, test_y)
            logger([
                ('step', n),
                ('train_loss', train_loss),
                ('test_loss', test_loss)])

    logger.close('step')
    return model
        

def main():
    batch_size = 20
    input_size = 3
    hidden_size = 5
    output_size = 1
    trained_model = step_based_training_loop(
        model=FullyConnectedModel([input_size, output_size]),
        loss=mean_squared_loss,
        train_gen=tree_height_generator(batch_size),
        test_gen=tree_height_generator(batch_size),
    )

if __name__ == '__main__': 
    main()
