import numpy as np

def gen_test_data(times, datagen)

    formats = cls.legitimate_input()

    for i in range(times):

        chosen_format = datagen
        chosen_inputs = chosen_format[0].copy()
        chosen_outputs = chosen_format[1].copy()

        inputs = [ np.rand(x) for x in chosen_inputs ]
        ouputs = [ np.rand(chosen_outputs) ]
        yield [ inputs, outputs ]
