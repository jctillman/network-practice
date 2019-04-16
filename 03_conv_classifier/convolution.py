import numpy as np

from graph_components import (
    GraphComponent,
    MatrixMult,
    MatrixAdd,
    Relu,
    SequentialModel)

from utils import (
    output_dims,
    to_patches_2d,
    from_patches_2d)

class ConvolutionMult(GraphComponent):

    def __init__(self, input_shape, kernel_size, output_c):
        self.input_shape = input_shape
        self.input_channels = input_shape[2]
        self.k_size = kernel_size
        self.output_channels = output_c

        assert len(input_shape) == 3
        self.width = self.input_shape[0]
        self.height = self.input_shape[1]
        
        output_dim = output_dims(input_shape, kernel_size, 1) 
        self.output_dim = output_dim
        internal_size = kernel_size * kernel_size * output_c
        self.kernel_shape = (
            kernel_size * kernel_size * self.input_channels,
            self.output_channels) 
        super().__init__()

    def _get_params(self):
        return {
            'W': np.random.rand(*self.kernel_shape) - 0.5
        }

    def forward_single(self, single):
        transformed = to_patches_2d(
            single, size=self.k_size, stride=1)
        
        self.transformed_input.append(transformed)

        multiplied = np.matmul(transformed, self.params['W'])

        back_to_shape = from_patches_2d(multiplied,
            self.output_dim[0], self.output_dim[1],
            self.output_channels)

        return back_to_shape

    def forward(self, input):
        self.input = input
        results = []
        self.transformed_input = []
        for i in range(input.shape[0]):
            forward_result = self.forward_single(input[i])
            results.append(forward_result)
        self.output = np.array(results)
        return self.output

    def backward_single(self, i, single_deriv):
        
        transformed = to_patches_2d(
            single_deriv, size=1, stride=1)

        self.derivs['W'] = self.derivs['W'] + np.matmul(
            self.transformed_input[i].T, transformed) * 1

        self.errors[i] = from_patches_2d(
            np.matmul(transformed, self.params['W'].T),
            self.width,
            self.height,
            self.input_channels,
            size=self.k_size) 


    def backward(self, derivative):
        results = []
        self.derivs['W'] = np.zeros(self.params['W'].shape)
        self.errors = np.zeros(self.input.shape)
        for i in range(derivative.shape[0]):
            indiv_deriv = derivative[i]
            backward_result = self.backward_single(i, indiv_deriv)
            results.append(backward_result)

        return self.errors


class Convolution(SequentialModel):
    
    def __init__(self, input_shape, kernel_size, output_c):

        self.modules = [
            ConvolutionMult(input_shape, kernel_size, output_c),
            MatrixAdd((input_shape[0] - kernel_size + 1, input_shape[1] - kernel_size + 1, output_c)),
            Relu()
        ]

class Pad(GraphComponent):
    
    def __init__(self, pad_amount):
        self.pad_amount = pad_amount
        super().__init__()

    def forward(self, input):
        assert len(input.shape) == 4
        s = input.shape
        pa = self.pad_amount
        zeros = np.zeros((s[0], s[1] + pa * 2, s[2] + pa * 2, s[3]))
        zeros[:,pa:-pa, pa:-pa,:] = input
        return zeros

    def backward(self, derivative):
        pa = self.pad_amount
        return derivative[:,pa:-pa,pa:-pa,:]

class MaxPool(GraphComponent):

    def __init__(self, size, strides=None):

        #assert size > 1

        self.size = size
        if strides is None:
            self.strides = size
        else:
            self.strides = strides
        super().__init__()

    def forward(self, input):

        shape = input.shape
        size = self.size
        bn, old_w, old_h, c = shape
        assert old_w % self.size == 0
        assert old_h % self.size == 0

        new_w = int(old_w / self.size)
        new_h = int(old_h / self.size)
        
        out = np.zeros((bn, new_w, new_h, c))
        switches = np.zeros((bn, old_w, old_h, c))

        assert np.array_equal(switches.shape, shape)

        for ii, i in enumerate(range(0, old_w, self.strides)):
            for jj, j in enumerate(range(0, old_w, self.strides)):
                to_examine = input[:, i:i+size,j:j+size, :] 
                reshaped = to_examine.reshape(bn, -1, c)
                max_point = np.amax(reshaped, axis=1)
                out[:, ii, jj, :] = max_point
                switch = np.where(
                    to_examine == max_point.reshape(bn, 1, 1, c),
                    np.ones(to_examine.shape),
                    np.zeros(to_examine.shape))
                
                assert np.array_equal(to_examine.shape, switch.shape)
                switches[:, i:i+size, j:j+size, :] = switch

        self.switches = switches

        return out

    def backward(self, derivative):

        return np.multiply(
            self.switches,
            np.repeat(np.repeat(derivative, self.size, axis=1), self.size, axis=2))





