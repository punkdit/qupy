#!/usr/bin/env python3

from math import sqrt
from random import choice, randint, seed, shuffle
from functools import reduce
from operator import mul, matmul, add

import numpy

from qupy import scalar
#scalar.use_reals()

from qupy.scalar import EPSILON, MAX_GATE_SIZE

from qupy.abstract import Space
from qupy.dense import Qu, Gate, Vector
from qupy.dense import bitvec
from qupy.argv import argv
from qupy.util import mulclose, show_spec


I, X, Z, H = Gate.I, Gate.X, Gate.Z, Gate.H
S, T = Gate.S, Gate.T # S aka P
SWAP = Gate.SWAP
CX = Gate.CN

def find(op, ops):
    for op1 in ops:
        #if numpy.abs(op-op1).sum() < EPSILON:
        if op == op1:
            return True
    return False


def inv(A):
    A = A.v
    a, b = A[0]
    c, d = A[1]
    r = 1. / (a*d - b*c)
    v = numpy.array([[d, -b], [-c, a]])*r
    B = Qu((2, 2), 'ud', v)
    return B



def test():

    C1 = mulclose([X, Z])
    C2 = mulclose([H, Z])
    C3 = mulclose([H, S])
    assert S*S == Z

    print(len(C1))
    print(len(C2))
    print(len(C3))

    for g in C3:
        ginv = inv(g)
        assert g*ginv == I

    for g in C3:
        ginv = inv(g)
        assert g*ginv == I
        for h in C1:
            k = g * h * ginv
            assert find(k, C2) # fail...


if __name__ == "__main__":

    name = argv.next() or "test"

    fn = eval(name)
    fn()


