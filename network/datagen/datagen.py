import pandas as pd
import numpy as np
import csv
from random import random

import tensorflow as tf
mnist = tf.keras.datasets.mnist

def mapper(fnc, batch_size, input_name, output_name):
    yielder = fnc(batch_size)
    def gen():
        x, y = next(yielder)
        return { input_name: x }, { output_name: y }
    return gen
    

def to_one_hot(mx, num):
    n = np.zeros((mx))
    n[num] = 1
    return n

def mnist_generator(batch_size, train=True):

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    xs, ys = None, None
    if train == True:
        xs = x_train / 255 - 0.5
        ys = y_train
    else:
        xs = x_test / 255 - 0.5
        ys = y_test

    while True:
        b_xs = []
        b_ys = []
        for _ in range(batch_size):
            pull = np.random.randint(0, len(xs))
            b_xs.append(xs[pull])
            b_ys.append(ys[pull])
        b_xs = np.array(b_xs) 
        b_ys = np.array(b_ys)
        
        b_ys = np.array(
            list(map(lambda x: to_one_hot(10, x), b_ys)))

        yield b_xs.reshape((-1, 28 * 28)), b_ys

    assert False

def stupid_fsm():
    BN = 4
    T = 15
    NUM = 3
    forward_data = np.zeros((BN, T, NUM))
    for i in range(BN):
        prior = 'A' if random() > 0.1 else 'C'
        for ii in range(T):
            if prior == 'A':
                if random() > 0.1:
                    next = 'B'
                else:
                    next = 'C'
            if prior == 'B':
                if random() > 0.1:
                    next = 'C'
                else:
                    next = 'B'
            if prior == 'C':
                if random() > 0.9:
                    next = 'A'
                else:
                    next = 'B'
            if next == 'A':
                forward_data[i,ii, 0] = 1
            if next == 'B':
                forward_data[i, ii, 1] = 1
            if next == 'C':
                forward_data[i, ii, 2] = 1
            prior = next
    return forward_data

def alt_patterns():
    BN = 4
    T = 15
    NUM = 3
    forward_data = np.zeros((BN, T, NUM))
    for i in range(BN):
        which = random() > 0.5
        for ii in range(T):
            if which:
                if ii % 3 == 0:
                    forward_data[i,ii, 0] = 1
                else:
                    forward_data[i,ii, 1] = 1
            else:
                if (ii + 1) % 3 == 0:
                    forward_data[i,ii, 1] = 1
                else:
                    forward_data[i,ii, 2] = 1
    return forward_data
            

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
