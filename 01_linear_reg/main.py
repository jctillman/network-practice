import numpy as np

class CoinFlipGenerator:
    
    '''
    Going to return data_x, data_y

    data_x is of [bs, sequence_size] dimensions
    data_y is of [bs, len(probs) ] dimensions
    '''

    def __init__(self, probs, sequence_size, batch_size):
        self.probs = probs
        self.sequence_size = sequence_size
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        choice = np.random.choice
        indices = range(len(self.probs))
        
        coin_sequences = []
        coin_answers = []
        for coin_index in choice(indices, size=self.batch_size):
            
            # Generate flips
            sequence = []
            prob = self.probs[coin_index]
            for flip in range(self.sequence_size):
                flip_result = 1 if np.random.random() < prob else 0
                sequence.append(flip_result)
            coin_sequences.append(sequence)

            # Index of coin
            answer = [0] * len(self.probs)
            answer[coin_index] = 1
            coin_answers.append(answer)


        return np.array(coin_sequences), np.array(coin_answers)


class LinearClassificationModel(object):

    def __init__(self, input_size, output_size):

        self.input_size = input_size
        self.output_sie = output_size

        W = np.random.rand(input_size, output_size) - 0.5
        dW = np.zeros((input_size, output_size))
        self.params = { 'W': W }
        self.derivs = { 'W': dW }

    def forward(self, input):
        params = self.params
        out = np.dot(input, params['W'])
        out_exp = np.exp(out)
        out = out_exp / np.expand_dims(out_exp.sum(axis=1), axis=1)
        #print('out', out)
        return out


    def backward(self, derivative):
        params = self.params
        self.derivs['W'] = np.multiply(self.params['W'], derivative)

    def optimize(self):
        params = self.params
        derivs = self.derivs
        for k in params.keys():
            params[k] = params[k] - (derivs[k] * 0.01)

class MeanSquared:

    def __init__(self):
        pass

    def __call__(self, prediction, truth):
        assert prediction.shape == truth.shape

        loss = np.mean(np.power(truth - prediction, 2))
        deriv = np.sum(truth - prediction, 0)
        return loss, deriv

class DefaultLogger:
    def __init__(self):
        pass

class StepBasedTrainingLoop:

    def __init__(self,
                model=None,
                train_gen=None,
                test_gen=None,
                loss=MeanSquared(),
                total_steps=10,
                test_every=1,
                logger=DefaultLogger()):

        self.model = model
        self.train_gen = train_gen
        self.test_gen = test_gen
        self.loss = loss
        self.total_steps = total_steps
        self.test_every = test_every
        self.logger = logger
    
    def __call__(self):

        for n in range(self.total_steps):

            data_x, data_y = next(self.train_gen)
            #print(n)
            #print(data_x)
            #print(data_y)
            pred = self.model.forward(data_x)
            #print(pred, data_y)
            loss, deriv = self.loss(pred, data_y)
            print(loss)
            #print(loss, deriv)
            self.model.backward(deriv)
            self.model.optimize()

       
        


def main():
    flip_number = 20
    num_coins = 2
    batch_size = 30
    StepBasedTrainingLoop(
        model=LinearClassificationModel(flip_number, num_coins),
        train_gen=iter(CoinFlipGenerator([0.6, 0.2], flip_number, batch_size)),
        test_gen=iter(CoinFlipGenerator([0.6, 0.2], flip_number, batch_size)),
        total_steps=1000,
        test_every=1,
    )()

if __name__ == '__main__': 
    print("Running")
    main()
