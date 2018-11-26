#!/usr/bin/env python3

import math
from random import choice, randint, seed, shuffle
from functools import reduce
from operator import mul, matmul

import numpy

from qupy import scalar
scalar.use_reals()

from qupy.abstract import Space
from qupy.dense import Qu, Gate, Vector
from qupy.dense import genidx, bits, is_close, on, off, scalar
from qupy.dense import commutator, anticommutator


I, X, Z, H = Gate.I, Gate.X, Gate.Z, Gate.H
SWAP = Gate.SWAP

def main():

    III = I@I@I
    s0 = SWAP@I
    s1 = I@SWAP

    assert (s0*s0).is_close(III)
    assert (s1*s1).is_close(III)
    assert (s0*s1*s0).is_close(s1*s0*s1)

    items = s0.eigs()
    for val, vec in items:
        print(val, vec.shortstr())
        #print(val, vec)
        print((s0*vec).shortstr())
        print()
    


if __name__ == "__main__":

    main()


