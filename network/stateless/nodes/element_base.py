import numpy as np

class StatelessOp:

    @classmethod
    def forw(cls, inputs=None):
        raise Exception("Not implemented forward")

    @classmethod
    def back(cls, inputs=None, outputs=None, error=None):
        raise Exception("Not implemented backward")

    @classmethod
    def forward(cls, *arguments):
        assert len(arguments) == cls.input_num or cls.input_num is None
        return cls.forw(inputs = arguments)

    @classmethod
    def backward(cls, inputs=None, outputs=None, error=None):
        assert inputs is not None
        assert error is not None
        if outputs is None:
            outputs = cls.forw(inputs=inputs)
        results = cls.back(inputs=inputs, outputs=outputs, error=error)
        assert len(results) == len(inputs)

        # Slows down stuff a ton, don't think it's that important
        # except when testing new elements.
        #for i in range(len(results)):
        #    assert np.array_equal(results[i].shape, inputs[i].shape)
        return results
