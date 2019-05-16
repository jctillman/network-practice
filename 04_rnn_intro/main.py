
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

from graph_components import (
    GraphComponent,
    MatrixMult, MatrixAdd,
    Relu, Softmax, SequentialModel, Reshape)

from convolution import (
    Convolution, MaxPool, Pad)

from loss import mean_squared_loss

from datagen import mnist_generator

import tensorflow as tf
mnist = tf.keras.datasets.mnist

class DefaultLogger:

    def __init__(self, file_name):
        self.file_name = file_name
        self.info = {}


    def __call__(self, values):
        s = ''
        for name, value in values: 
            if name not in self.info:
                self.info[name] = []
            s = s + name + ': ' + str(value) + '\t\t'
            self.info[name].append(value)
        print(s)

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

class FullyConnectedModel(SequentialModel):

    def __init__(self):

        hidden_size = 30
        self.modules = [
            Reshape((28, 28), (28 * 28,)),
            MatrixMult([28 * 28, hidden_size]),
            MatrixAdd([hidden_size]),
            Relu(),
            MatrixMult([hidden_size, hidden_size]),
            MatrixAdd([hidden_size]),
            Relu(),
            MatrixMult([hidden_size, 10]),
            MatrixAdd([10]),
            Softmax(),
        ]

class Identity(GraphComponent):
    
    def forward(self, input):
        print(input[0])
        return input

    def backward(self, derivative):
        return derivative

class MnistModel(SequentialModel):

    def __init__(self):

        output_size = 10
        hidden_size = 12
        self.modules = [
            Reshape((28, 28), (28, 28, 1)),
            Convolution((28, 28, 1), 5, 8), # to 24
            MaxPool(2),  # to 12
            Pad(1), # to 14
            Convolution((14, 14, 8), 3, 16), #to 12
            MaxPool(2), # to 6
            Convolution((6, 6, 16), 3, 24), # to 4
            MaxPool(2), # to 2
            Reshape((2,2,24),(2 * 2 * 24,)),
            MatrixMult([2 * 2 * 24, hidden_size]),
            MatrixAdd([hidden_size]),
            Relu(),
            MatrixMult([hidden_size, output_size]),
            Softmax(),
        ]

def step_based_training_loop(
    model=None,
    loss=None,
    train_gen=None,
    test_gen=None,
    total_steps=10000,
    test_frequency=10,
    learning_rate=0.005,
    logger=DefaultLogger('untitled_run')):

    for n in range(total_steps):

        data_x, data_y = next(train_gen)
        pred = model.forward(data_x)
        train_loss, deriv = loss(prediction=pred, truth=data_y)
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
    batch_size = 10
    trained_model = step_based_training_loop(
        model=MnistModel(),
        loss=mean_squared_loss,
        test_frequency=10,
        train_gen=mnist_generator(batch_size, train=True),
        test_gen=mnist_generator(batch_size, train=False),
        logger=DefaultLogger('conv_run')
    )

if __name__ == '__main__': 
    main()
