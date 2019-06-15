import pandas as pd
import numpy as np
import csv

import tensorflow as tf
mnist = tf.keras.datasets.mnist

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
