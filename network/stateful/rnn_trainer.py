
import stateless.graph.graph_linked as gl
from stateless.graph.rnn_converter import to_rnn
import numpy as np


class RNNTrainer():

    def __init__(self, linked, initial_weights, initial_hidden, loss, optimizer):
        self.linked = to_rnn(linked)
        self.weights = initial_weights
        self.initial_hidden = initial_hidden
        self.optimizer = optimizer
        self.loss = loss

    def train_single(self, input_x, num = 0):

        results = self.linked.forw(input_x, self.weights, self.initial_hidden)
        total_loss, gradients = self.loss(input=input_x, prediction=results)
        if num % 20 == 0:
            print(num, ' loss ', total_loss)
        back = list(self.weights.keys()) + [
                x for x in
                self.linked.linked.dag.get_node_names()
                if 'prior_' in x ]

        backwards = self.linked.back(
            results,
            gradients,
            back)
        self.optimizer(self.weights, backwards)

    def test(self, input_x):
    
        results = self.linked.forw(input_x, self.weights, self.initial_hidden)
        total_loss, gradients = self.loss(input=input_x, prediction=results)
        return total_loss


    def train_batch(self, steps, batch_gen):
        for i in range(steps):
            input_x = batch_gen()
            self.train_single(input_x, i)

    def predict(self, num, initial, concretizer):
        slices = [initial]
        return self.linked.forw_create(
            num,
            initial,
            self.weights,
            self.initial_hidden,
            concretizer,
        )


