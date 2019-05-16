import numpy as np

from stateless_graph_base import StatelessOp
from utils import list_equals

class MatrixMult(StatelessOp):
    '''
    Lets say M x N times N x P
    So we get M x P

    To get M x N from that we do:
    M x P times (N x P).T

    To get N x P from that we do:
    (M x N).T and M x P
    '''
    input_num = 2
    @classmethod
    def forw(cls, inputs=None):
        assert len(inputs) == 2
        assert inputs[0].shape[1] == inputs[1].shape[0]
        return np.matmul(inputs[0], inputs[1])

    @classmethod
    def back(cls, inputs=None, outputs=None, error=None):
        firstError = np.matmul(error, inputs[1].T)
        secondError = np.matmul(inputs[0].T, error)
        return [ firstError, secondError]

class MatrixAddExact(StatelessOp):
    '''
    Takes two matrices of absolutely identical
    dimensions, and adds them.  And returns derivative.
    '''
    input_num = 2
    @classmethod
    def forw(cls, inputs=None):
        assert len(inputs) == 2
        assert list_equals(inputs[0].shape, inputs[1].shape) 
        return inputs[0] + inputs[1]

    @classmethod
    def back(cls, inputs=None, outputs=None, error=None):
        return [ error, error ]

class MatrixAdd(StatelessOp):
    '''
    Takes two matrices where the second is one
    less in the first dimension.  And returns them.
    '''
    input_num = 2
    @classmethod
    def forw(cls, inputs=None):
        assert len(inputs) == 2
        assert list_equals(inputs[0].shape[1:], inputs[1].shape) 
        return inputs[0] + inputs[1]

    @classmethod
    def back(cls, inputs=None, outputs=None, error=None):
        errorSmaller = np.sum(error, axis=0)
        return [ error, errorSmaller ]

class Relu(StatelessOp):

    input_num = 1
    @classmethod
    def forw(cls, inputs=None):
        assert len(inputs) == 1
        return np.clip(inputs[0], a_min=0, a_max=None)

    @classmethod
    def back(cls, inputs=None, outputs=None, error=None):
        return [ np.where(inputs[0] > 0, error, np.zeros(error.shape)) ]

class Exponent(StatelessOp):

    input_num = 1
    @classmethod
    def forw(cls, inputs=None):
        assert len(inputs) == 1
        return np.exp(inputs[0])

    @classmethod
    def back(cls, inputs=None, outputs=None, error=None):
        return [ np.multiply(outputs, error) ]

class Sigmoid(StatelessOp):

    input_num = 1
    @classmethod
    def forw(cls, inputs=None):
        assert len(inputs) == 1
        exp = np.exp(inputs[0])
        return np.divide(exp, exp+ + 1)

    @classmethod
    def back(cls, inputs=None, outputs=None, error=None):
        tmp = np.multiply(outputs, 1 - outputs)
        return [ np.multiply(tmp, error) ]

class Identity(StatelessOp):

    input_num = 1
    @classmethod
    def forw(cls, inputs=None):
        return inputs[0]

    @classmethod
    def back(cls, inputs=None, outputs=None, error=None):
        return [ error ]


class Concat(StatelessOp):
    
    input_num = 2
    @classmethod
    def forw(cls, inputs=None):
        assert len(inputs) == 2
        assert inputs[0].shape[0] == inputs[1].shape[0]
        return np.concatenate((inputs[0], inputs[1]), axis=1)

    @classmethod
    def back(cls, inputs=None, outputs=None, error=None):
        split_point = inputs[0].shape[1]
        return [ error[:,:split_point ], error[:,split_point:] ]

class Probabilize(StatelessOp):

    input_num = 1
    @classmethod
    def forw(cls, inputs = None):
        assert len(inputs) == 1
        assert np.all(inputs[0] >= 0)
        assert len(inputs[0].shape) == 2
        summed = np.sum(inputs[0], axis=1)
        expanded_summed = np.expand_dims(summed, 1)
        return np.divide(inputs[0], expanded_summed)

    @classmethod
    def back(cls, inputs=None, outputs=None, error=None):

        summed = np.sum(inputs[0], axis=1)
        output = outputs
        all_examples = []
        for i in range(output.shape[0]):
            ith_case = inputs[0][i]
            jacobian = []
            total = summed[i]
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

            result = np.multiply(np.array(jacobian), np.expand_dims(error[i], 0))

            all_examples.append(np.sum(np.array(result), axis=1))

        return [ np.array(all_examples) ]
