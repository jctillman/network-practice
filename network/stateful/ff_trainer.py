
import stateless.graph.graph_linked as gl


class FFTrainer():

    def __init__(self, linked, initial_weights, loss, optimizer):
        self.linked = linked
        self.weights = initial_weights
        self.optimizer = optimizer
        self.loss = loss

    def train_single(self, input_x, input_y):

        results = self.linked.forw({ **self.weights, **input_x })
        loss, gradient = self.loss(prediction=results, truth=input_y)
        print(loss)
        backward = self.linked.back(gradient, results, self.weights.keys())
        self.optimizer(self.weights, backward)

    def test_single(self, input_x, input_y):

        results = self.linked.forward({ **self.weights, **input_x })
        loss, gradient = self.loss(results, input_y)
        return loss

    def train_batch(self, steps, batch_gen):
        for i in range(steps):
            input_x, input_y = batch_gen()
            self.train_single(input_x, input_y)
