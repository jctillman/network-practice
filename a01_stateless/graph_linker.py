
import stateless_graph_components as sgc

from graph_linker_base import linker

Relu = linker(sgc.Relu)
MatrixMult = linker(sgc.MatrixMult)
MatrixAddExact = linker(sgc.MatrixAddExact)
MatrixAdd = linker(sgc.MatrixAdd)
Exponent = linker(sgc.Exponent)
Identity = linker(sgc.Identity)
Concat = linker(sgc.Concat)
Probabilize = linker(sgc.Probabilize)
Sigmoid = linker(sgc.Sigmoid)
