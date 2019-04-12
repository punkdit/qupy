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


# argh... make this a module?
I, X, Z, H = Gate.I, Gate.X, Gate.Z, Gate.H
S, T = Gate.S, Gate.T # S aka P
SWAP = Gate.SWAP
CX = Gate.CN


def main():

    ir2 = 1./sqrt(2)
    zero = bitvec(0)
    one = bitvec(1)
    plus = ir2 * (bitvec(0) + bitvec(1))
    minus = ir2 * (bitvec(0) - bitvec(1))
    bell = ir2 * (bitvec(0, 0) + bitvec(1, 1))

    CZ = Z.control()
    assert CX == X.control()

    print( (I@H) * CZ * (I@H) == CX )
    print( (I@H) * CX * (I@H) == CZ )
    print( CX * (I@H) == (I@H) * CZ )
    print( CZ * (I@H) == (I@H) * CX )

    # lhs -----------------------------

    A = CZ @ I
    B = H @ H @ I
    C = I @ CZ
    D = I @ I @ H
    E = Z.control(2)

    op = E * D * C * B * A

    lhs = op * ( I @ bell )

    #print(lhs.shortstr())

    # rhs -----------------------------

    assert (SWAP*SWAP) == I@I
    A = (I@SWAP) * (SWAP@I)
    rhs = A * (I @ plus @ plus)

    #print(rhs.shortstr())

    assert lhs == rhs




if __name__ == "__main__":

    main()


