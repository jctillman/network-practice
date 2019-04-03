
import numpy as np

def tree_height_generator(batch_size):
    '''
    Going to return data_x, data_y
    data_x is of [bs, 3] dimensions
    data_y is of [bs, 1] dimensions
    '''
    def get_tuples():
        x_grow_time = np.random.random()
        x_is_transplant = 1 if np.random.random() > 0.5 else 0
        inp = [x_grow_time, x_is_transplant ]
        out = [x_grow_time * 2.3 + x_is_transplant + 0.5]
        return (inp, out)

    while True:
        inp, out = zip(*[ get_tuples() for _ in range(batch_size) ])
        yield np.array(inp), np.array(out)


def mean_squared_loss(prediction=None, truth=None):
    assert prediction is not None
    assert truth is not None
    assert prediction.shape == truth.shape
    loss = 0.5 * np.sum(np.power(prediction - truth, 2))
    deriv = np.sum(prediction - truth, 0)
    return loss, deriv

def default_logger(step, train_loss, test_loss):
    print(step, train_loss, test_loss)

class LinearRegressionModel(object):

    def __init__(self, input_size, output_size):

        self.input_size = input_size
        self.output_size = output_size

        W = np.random.rand(input_size, output_size)
        B = np.zeros((output_size))
        dW = np.zeros((input_size, output_size))
        dB = np.zeros((output_size))
        self.params = { 'W': W, 'B': B }
        self.derivs = { 'W': dW, 'B': dB }
        self.input = None

    def forward(self, input):
        assert input.shape[1] == self.input_size
        self.input = input
        para = self.params
        return np.matmul(input, para['W']) + para['B']


    def backward(self, derivative):
        #dWA = np.zeros(self.derivs['W'].shape)
        #for a in range(len(self.input)):
        #    inp = np.expand_dims(self.input[a], axis=1)
        #    chang = np.multiply(inp, derivative)
        #    dWA = dWA + chang 

        accum_input = np.expand_dims(np.sum(self.input, 0), axis=1)
        times_deriv = np.multiply(derivative, accum_input)
        
        self.derivs['W'] = times_deriv
        self.derivs['B'] = derivative
        assert self.derivs['W'].shape == self.params['W'].shape
        assert self.derivs['B'].shape == self.params['B'].shape

        
        expanded_deriv = np.expand_dims(derivative, axis=1)
        error = np.matmul(self.params['W'], expanded_deriv)
        #print(derivative.shape, error.shape, self.input.shape)
        #assert error.shape == self.input.shape
        return np.squeeze(error)


    def optimize(self, learning_rate):
        params = self.params
        derivs = self.derivs
        for k in ['W', 'B']:
            assert derivs[k].shape == params[k].shape
            params[k] = params[k] - (derivs[k] * learning_rate)

def step_based_training_loop(
    model=None,
    loss=None,
    train_gen=None,
    test_gen=None,
    total_steps=10000,
    test_frequency=1000,
    learning_rate=0.0001,
    logger=default_logger):

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
            logger(n, train_loss, test_loss)

    return model
        

def main():
    batch_size = 50
    input_size = 2
    output_size = 1
    trained_model = step_based_training_loop(
        model=LinearRegressionModel(input_size, output_size),
        loss=mean_squared_loss,
        train_gen=tree_height_generator(batch_size),
        test_gen=tree_height_generator(batch_size),
    )

if __name__ == '__main__': 
    main()
