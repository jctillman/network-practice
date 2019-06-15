
import numpy as np
from numpy.random import random, rand

from stateless.nodes.elements import (
    Relu,
    Sigmoid,
    Exponent,
    Identity,
    Probabilize,
    Concat,
    MatrixAdd,
    MatrixMult,
    MatrixAddExact)

from stateless.utils.utils_for_test import (
    shape_from_format,
    gen_test_data,
    do_test_runs,
    do_test_derivs)


def test_MatrixMult():
    shape_generator = shape_from_format([['N','M'],['M','P']],['N','P'])
    do_test_runs(MatrixMult, shape_generator)
    do_test_derivs(MatrixMult, shape_generator)

def test_MatrixAddExact():
    shape_generator = shape_from_format([['N','P'],['N','P']],['N','P'])
    do_test_runs(MatrixAddExact, shape_generator)
    do_test_derivs(MatrixAddExact, shape_generator)
    
def test_MatrixAdd():
    shape_generator = shape_from_format([['N','P'],['P']],['N','P'])
    do_test_runs(MatrixAdd, shape_generator)
    do_test_derivs(MatrixAdd, shape_generator)

def test_Relu():
    shape_generator = shape_from_format([['N','P']],['N','P'])
    do_test_runs(Relu, shape_generator)
    do_test_derivs(Relu, shape_generator)

def test_Sigmoid():
    shape_generator = shape_from_format([['N','P']],['N','P'])
    do_test_runs(Sigmoid, shape_generator)
    do_test_derivs(Sigmoid, shape_generator)

def test_Exponent():
    shape_generator = shape_from_format([['N','P']],['N','P'])
    do_test_runs(Exponent, shape_generator)
    do_test_derivs(Exponent, shape_generator)

def test_Probabilize():
    shape_generator = shape_from_format([['N','P']],['N','P'])
    do_test_runs(Probabilize, shape_generator)
    do_test_derivs(Probabilize, shape_generator, permissible_skip_ratio=0.6)

def test_Identity():
    shape_generator = shape_from_format([['N','P']],['N','P'])
    do_test_runs(Identity, shape_generator)
    do_test_derivs(Identity, shape_generator)

def test_Concat():
    shape_generator = shape_from_format([['N','P'],['N', 'Q']],['N','P+Q'])
    do_test_runs(Concat, shape_generator)
    do_test_derivs(Concat, shape_generator)


