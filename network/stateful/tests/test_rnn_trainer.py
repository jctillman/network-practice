import pytest

from stateless.utils.dag import Dag
from stateless.utils.utils import list_equals
from stateless.loss.loss import mean_squared_loss, applied, running_rnn_loss
from stateless.optimizer.sgd import get_sgd_optimizer
from stateful.rnn_trainer import RNNTrainer
from numpy.random import rand
import numpy as np

from datagen.datagen import tree_height_generator, mnist_generator, mapper
from datagen.datagen import stupid_fsm, alt_patterns, text_generator, to_one_hot, one_hot_to_text


from stateless.graph.graph_linked import (
    Relu,
    LeakyRelu,
    Exponent,
    Identity,
    Input,
    Parameter,
    Probabilize,
    Concat,
    TanH,
    Sigmoid,
    ElementwiseMult,
    MatrixAdd,
    MatrixMult,
    MatrixAddExact)

from stateless.loss.loss import mean_squared_loss

all_linkers = [
    Relu,
    Exponent,
    Identity,
    Input,
    Parameter,
    Probabilize,
    Concat,
    MatrixAdd,
    MatrixMult,
    MatrixAddExact]

def test_rnn_works_simple():

    i = Identity([], name='input')

    def get_recursive_layer(prior, layer_name, weight_name, bias_name):
        fcw1 = Identity([], name=weight_name)
        fcb1 = Identity([], name=bias_name)
        ii = Identity([], name='prior_' + layer_name)
        joined = Concat([prior, ii])
        multed = MatrixMult([joined, fcw1])
        added = MatrixAdd([multed, fcb1])
        h1 = Relu(added, name=layer_name+'_internal')
        h11 = Identity([h1], name=layer_name)
        return h1
    
    h1 = get_recursive_layer(i, 'h1', 'fc_w1', 'fc_b1')
    fcw2 = Identity([], name='fc_w2')
    fcb2 = Identity([], name='fc_b2')
    output = (
        Relu(
            MatrixAdd([MatrixMult([h1, fcw2]), fcb2]),
        name='output')
    )

    BN = 4
    NUM = 3    
    H_SIZE = 6
    weights = {
        'fc_w1': 0.05 * np.random.rand(3 + H_SIZE, H_SIZE),
        'fc_b1': 0.05 * np.random.rand(H_SIZE),
        'fc_w2': 0.05 * np.random.rand(H_SIZE, NUM),
        'fc_b2': 0.05 * np.random.rand(NUM),
    }
    optimizer = get_sgd_optimizer(0.0025)
    trainer = RNNTrainer(
        output, 
        weights,
        { 'h1': np.zeros((BN, H_SIZE))},
        running_rnn_loss('input', 'output', mean_squared_loss),
        optimizer)

    def batch_gen():
        return { 'input': stupid_fsm() }

    test_batch = batch_gen()
    initial_loss = trainer.test(test_batch)
    trainer.train_batch(300, batch_gen)
    
    other_loss = trainer.test(test_batch)
    
    assert other_loss * 3 < initial_loss
    trainer.initial_hidden = { 'h1': np.zeros((1, H_SIZE)) }
    num = 20
    initial = np.array([[1,0,0]])
    
    def concretizer(val):
        m = np.random.choice(np.array([0, 1, 2]), p=val['output'][0] / sum(val['output'][0]))
        ret = np.array([0,0,0])
        ret[m] = 1
        return { **val, 'input': np.array([ret]) }

    predicted = trainer.predict(num, {
        'input': initial
    }, concretizer)

def test_rnn_multistep():
    
    i = Identity([], name='input')


    def get_recursive_layer(prior, layer_name, weight_name, bias_name):
        fcw1 = Identity([], name=weight_name)
        fcb1 = Identity([], name=bias_name)
        ii = Identity([], name='prior_' + layer_name)
        joined = Concat([prior, ii])
        multed = MatrixMult([joined, fcw1])
        added = MatrixAdd([multed, fcb1])
        h1 = Relu(added, name=layer_name+'_internal')
        h11 = Identity([h1], name=layer_name)
        return h1

    h1 = get_recursive_layer(i, 'h1', 'fc_w1', 'fc_b1')
    h2 = get_recursive_layer(h1, 'h2', 'fc_w2', 'fc_b2')

    fcw3 = Identity([], name='fc_w3')
    fcb3 = Identity([], name='fc_b3')

    output = (
        Exponent(
            MatrixAdd([MatrixMult([h2, fcw3]), fcb3]), name='output')
    )

    BN = 4
    T = 15
    NUM = 3    
    H_SIZE = 13
    weights = {
        'fc_w1': 0.2 * (np.random.rand(3 + H_SIZE, H_SIZE) - 0.5),
        'fc_b1': 0.2 * (np.random.rand(H_SIZE) - 0.5),
        'fc_w2': 0.2 * (np.random.rand(H_SIZE + H_SIZE, H_SIZE) - 0.5),
        'fc_b2': 0.2 * (np.random.rand(H_SIZE) - 0.5),
        'fc_w3': 0.2 * (np.random.rand(H_SIZE, NUM) - 0.5),
        'fc_b3': 0.2 * (np.random.rand(NUM) - 0.5),
    }

    optimizer = get_sgd_optimizer(0.004)
    trainer = RNNTrainer(
        output, 
        weights,
        { 'h1': np.zeros((BN, H_SIZE)), 'h2': np.zeros((BN, H_SIZE)) },
        running_rnn_loss('input', 'output', mean_squared_loss),
        optimizer)
    
    def batch_gen():
        nmn = alt_patterns()
        return { 'input': nmn }

    test_batch = batch_gen()
    initial_loss = trainer.test(test_batch)
    trainer.train_batch(500, batch_gen)
    final_loss = trainer.test(test_batch)
    assert final_loss * 3 < initial_loss

    trainer.initial_hidden = { 'h1': np.zeros((1, H_SIZE)), 'h2': np.zeros((1, H_SIZE)) }

    num = 20
    initial = np.array([[0,0,1]])
    
    def concretizer(val):
        #m = np.random.choice(np.array([0, 1, 2]), p=val['output'][0] / sum(val['output'][0]))
        print(val['output'])
        m = np.argmax(val['output'])
        ret = np.array([0,0,0])
        ret[m] = 1
        return { **val, 'input': np.array([ret]) }

    predicted = trainer.predict(num, {
        'input': initial
    }, concretizer)

    print([x['input'] for x in predicted])


def test_rnn_lstm():
    
    i = Identity([], name='input')

    def get_lstm_layer(prior_layer, layer_name):

        # OUTPUT_DIM == C_DIM
        # INPUT_DIM == C_DIM

        # [BN x OUTPUT_DIM]
        prior_ht = Identity([], name='prior_' + layer_name + '_ht')
        # [BN x (OUTPUT_DIM + INPUT_DIM)]
        with_prior_ht = Concat([prior_layer, prior_ht])

        # [(OUTPUT_DIM + INPUT_DIM) x C_DIM ]
        ft_w = Identity([], name=layer_name + '_ft_w')
        # C_DIM ]
        ft_b = Identity([], name=layer_name + '_ft_b')
        # [BN x C_DIM]
        ft_mult = MatrixMult([with_prior_ht, ft_w])
        # [ BN x C_DIM ]
        ft = Sigmoid(MatrixAdd([ft_mult, ft_b]))

        # [(OUTPUT_DIM + INPUT_DIM) x C_DIM ]
        it_w = Identity([], name=layer_name + '_it_w')
        # [ C_DIM ]
        it_b = Identity([], name=layer_name + '_it_b')
        # [ BN x C_DIM ]
        it_mult = MatrixMult([with_prior_ht, it_w])
        # [ BN x C_DIM ]
        it = Sigmoid(MatrixAdd([it_mult, it_b]))

        # [(OUTPUT_DIM + INPUT_DIM) x C_DIM ]
        delta_c_w = Identity([], name=layer_name + '_delta_c_w')
        # [ C_DIM ]
        delta_c_b = Identity([], name=layer_name + '_delta_c_b')
        # [ BN x C_DIM ]
        delta_c_mult = MatrixMult([with_prior_ht, delta_c_w])
        # [ BN x C_DIM ]
        delta_c = TanH(MatrixAdd([delta_c_mult, delta_c_b]))

        prior_ct = Identity([], name='prior_' + layer_name + '_ct')

        # [ BN, C_DIM ] and [ BN, C_DIM ]
        ct_after_forget = ElementwiseMult([prior_ct, ft])
        # [ BN, C_DIM ] and [ BN, C_DIM ]
        ct = MatrixAddExact([ct_after_forget, ElementwiseMult([delta_c, it])])
        ct_pass = Identity([ct], name=layer_name + '_ct')

        # [ (OUTPUT_DIM + INPUT_DIM), OUTPUT_DIM ]
        output_c_w = Identity([], name=layer_name + '_output_c_w')
        # [ OUTPUT_DIM ]
        output_c_b = Identity([], name=layer_name + '_output_c_b')
        # [ BN, OUTPUT_DIM ]
        output_mult = MatrixMult([with_prior_ht, output_c_w])
        # [ BN, OUTPUT_DIM ]
        output_before_cond = Sigmoid(MatrixAdd([output_mult, output_c_b]))

        # [ BN, OUTPUT_DIM ] and [ BN, C_DIM ] so OUTPUT_DIM == C_DIM
        output = ElementwiseMult([output_before_cond, ct])
        output_pass = Identity([output], name=layer_name+'_ht')

        return output

    l1_name =  'lstm_layer_1'
    l2_name = 'lstm_layer_2'
    h1 = get_lstm_layer(i, l1_name)
    #h2 = get_lstm_layer(h1, l2_name)

    output = Identity([h1], name='output')

    T = 80
    vocab = ' abcdefghijklmnopqrstuvwxyz.,?;:!1234567890'
    INPUT_DIM = len(vocab)
    OUTPUT_DIM = INPUT_DIM
    C_DIM = INPUT_DIM

    weights = {
        l1_name + '_ft_w': 0.2 * (np.random.rand(INPUT_DIM + OUTPUT_DIM, C_DIM) - 0.5),
        l1_name + '_ft_b': 0.2 * (np.random.rand(C_DIM) - 0.5),

        l1_name + '_it_w': 0.2 * (np.random.rand(INPUT_DIM + OUTPUT_DIM, C_DIM) - 0.5),
        l1_name + '_it_b': 0.2 * (np.random.rand(C_DIM) - 0.5),

        l1_name + '_delta_c_w': 0.2 * (np.random.rand(INPUT_DIM + OUTPUT_DIM, C_DIM) - 0.5),
        l1_name + '_delta_c_b': 0.2 * (np.random.rand(C_DIM) - 0.5),

        l1_name + '_output_c_w': 0.2 * (np.random.rand(INPUT_DIM + OUTPUT_DIM, C_DIM) - 0.5),
        l1_name + '_output_c_b': 0.2 * (np.random.rand(C_DIM) - 0.5),

    }

    BN = 4
    T = 15
    NUM = 3    
    H_SIZE = 13

    optimizer = get_sgd_optimizer(0.001)
    trainer = RNNTrainer(
        output, 
        weights,
        {
            'lstm_layer_1_ht': np.zeros((BN, OUTPUT_DIM)),
            'lstm_layer_1_ct': np.zeros((BN, C_DIM)),
        },
        running_rnn_loss('input', 'output', mean_squared_loss),
        optimizer)

    gen = text_generator(4)
    def batch_gen():
        nmn = next(gen)
        return { 'input': nmn }

    test_batch = batch_gen()
    initial_loss = trainer.test(test_batch)
    trainer.train_batch(500, batch_gen)
    final_loss = trainer.test(test_batch)
    assert final_loss < initial_loss

    '''
    trainer.initial_hidden = {
        'lstm_layer_1_ht': np.zeros((1, OUTPUT_DIM)),
        'lstm_layer_1_ct': np.zeros((1, C_DIM)),
    }

    num = 80

    initial = np.array([to_one_hot(43,40)])
    
    for i in range(1, 50):

        def concretizer(val):
            m = np.random.choice(
                np.arange(43),
                p=np.exp(val['output'][0] * i) / sum(np.exp(val['output'][0] * i)))
            #m = np.argmax(val['output'])
            ret = np.zeros((43))
            ret[m] = 1
            return { **val, 'input': np.array([ret]) }

        predicted = trainer.predict(num, {
            'input': initial
        }, concretizer)

        print(''.join([ one_hot_to_text(vocab, x['input']) for x in predicted]))
    '''