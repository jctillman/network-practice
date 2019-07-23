import numpy as np

from stateless.nodes.element_base import StatelessOp
from stateless.utils.utils import list_equals

# Better to error fast than to let errors persist.
np.seterr(all='raise')


class ElementwiseMult(StatelessOp):
    '''
    Takes N x P and N x P and returns N x P
    '''
    input_num = 2
    @classmethod
    def forw(cls, inputs=None):
        if inputs[0].shape[0] != inputs[1].shape[0] or inputs[0].shape[1] != inputs[1].shape[1]:
            print(inputs[0].shape, inputs[1].shape)
            assert False
        return np.multiply(inputs[0], inputs[1])

    @classmethod
    def back(cls, inputs=None, outputs=None, error=None):
        first_err = np.multiply(error, inputs[1])
        second_err = np.multiply(inputs[0], error)
        return [ first_err , second_err]


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
        if not inputs[0].shape[1] == inputs[1].shape[0]:
            print("Must be multipliable: ", inputs[0].shape, inputs[1].shape)
            assert False
        return np.matmul(inputs[0], inputs[1])

    @classmethod
    def back(cls, inputs=None, outputs=None, error=None):
        first_err = np.matmul(error, inputs[1].T)
        second_err = np.matmul(inputs[0].T, error)
        return [ first_err, second_err]


class MatrixAddExact(StatelessOp):
    '''
    Takes two matrices of absolutely identical
    dimensions, and adds them.  And returns derivative.
    '''
    input_num = 2
    @classmethod
    def forw(cls, inputs=None):
        if not list_equals(inputs[0].shape, inputs[1].shape):
            print("Must equal exactly: ", inputs[0].shape, inputs[1].shape)
            assert False
        return inputs[0] + inputs[1]

    @classmethod
    def back(cls, inputs=None, outputs=None, error=None):
        return [ error, error ]


class MatrixAdd(StatelessOp):
    '''
    Takes two matrices where the second is one
    less in the first dimension.  Adds and
    returns them.
    '''
    input_num = 2
    @classmethod
    def forw(cls, inputs=None):
        assert list_equals(inputs[0].shape[1:], inputs[1].shape) 
        return inputs[0] + inputs[1]

    @classmethod
    def back(cls, inputs=None, outputs=None, error=None):
        errorForSmaller = np.sum(error, axis=0)
        return [ error, errorForSmaller ]


class Relu(StatelessOp):
    '''
    Takes a matrix and clips to be greater than 0
    '''
    input_num = 1
    @classmethod
    def forw(cls, inputs=None):
        return np.clip(inputs[0], a_min=0, a_max=None)

    @classmethod
    def back(cls, inputs=None, outputs=None, error=None):
        return [ np.where(inputs[0] > 0, error, np.zeros(error.shape)) ]

class LeakyRelu(StatelessOp):
    '''
    Takes a matrix and clips to be greater than 0
    '''
    input_num = 1
    @classmethod
    def forw(cls, inputs=None):
        return np.where(inputs[0] > 0, inputs[0], inputs[0] * 0.01 )

    @classmethod
    def back(cls, inputs=None, outputs=None, error=None):
        return [ np.where(inputs[0] > 0, error, error * 0.01 ) ]


MAX_EXP = 15
class Exponent(StatelessOp):
    '''
    Elementwise raising e by element.
    '''
    input_num = 1
    
    @classmethod
    def forw(cls, inputs=None):
        return np.exp(np.clip(inputs[0], a_min=-MAX_EXP, a_max=MAX_EXP))

    @classmethod
    def back(cls, inputs=None, outputs=None, error=None):
        return [ np.multiply(outputs, error) ]

class TanH(StatelessOp):
    '''
    Elementwise applies sigmoid
    '''
    input_num = 1
    @classmethod
    def forw(cls, inputs=None):
        return np.tanh(inputs[0])

    @classmethod
    def back(cls, inputs=None, outputs=None, error=None):
        tmp = 1 - np.power(outputs, 2)
        return [ np.multiply(tmp, error) ]

class Sigmoid(StatelessOp):
    '''
    Elementwise applies sigmoid
    '''
    input_num = 1
    @classmethod
    def forw(cls, inputs=None):
        exp = np.exp(inputs[0])
        return np.divide(exp, exp+ + 1)

    @classmethod
    def back(cls, inputs=None, outputs=None, error=None):
        tmp = np.multiply(outputs, 1 - outputs)
        return [ np.multiply(tmp, error) ]


class Identity(StatelessOp):
    '''
    Elementwise identity.
    '''
    input_num = 1
    @classmethod
    def forw(cls, inputs=None):
        return inputs[0]

    @classmethod
    def back(cls, inputs=None, outputs=None, error=None):
        return [ error ]


class Concat(StatelessOp):
    '''
    Concats along the 1st axis.
    '''
    input_num = 2
    @classmethod
    def forw(cls, inputs=None):
        if not inputs[0].shape[0] == inputs[1].shape[0]:
            print(inputs[0].shape, inputs[1].shape)
            assert False
        return np.concatenate((inputs[0], inputs[1]), axis=1)

    @classmethod
    def back(cls, inputs=None, outputs=None, error=None):
        split_point = inputs[0].shape[1]
        return [ error[:,:split_point ], error[:,split_point:] ]


class Probabilize(StatelessOp):
    '''
    Given 2d matrix, returns second axis
    as probility distribution.
    Requires inputs above 0.
    '''
    input_num = 1
    @classmethod
    def forw(cls, inputs = None):
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
