
import stateless.nodes.elements as els

from stateless.graph.graph_linker import linker

Relu = linker(els.Relu)
LeakyRelu = linker(els.LeakyRelu)
MatrixMult = linker(els.MatrixMult)
MatrixAddExact = linker(els.MatrixAddExact)
ElementwiseMult = linker(els.ElementwiseMult)
TanH = linker(els.TanH)
MatrixAdd = linker(els.MatrixAdd)
Exponent = linker(els.Exponent)
Identity = linker(els.Identity)
Concat = linker(els.Concat)
Probabilize = linker(els.Probabilize)
Sigmoid = linker(els.Sigmoid)

class Input(Identity):
    def __init__(self, name):
        super().__init__([], name)

class Prior(Identity):
    def __init__(self, name):
        super().__init__([], 'prior_' + name)

class Parameter(Identity):
    def __init__(self, name):
        super().__init__([], name)

