
import stateless.nodes.elements as els

from stateless.graph.graph_linker import linker

Relu = linker(els.Relu)
MatrixMult = linker(els.MatrixMult)
MatrixAddExact = linker(els.MatrixAddExact)
MatrixAdd = linker(els.MatrixAdd)
Exponent = linker(els.Exponent)
Identity = linker(els.Identity)
Concat = linker(els.Concat)
Probabilize = linker(els.Probabilize)
Sigmoid = linker(els.Sigmoid)
