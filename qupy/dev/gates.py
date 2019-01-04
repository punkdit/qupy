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


# argh... make this a module?
I, X, Z, H = Gate.I, Gate.X, Gate.Z, Gate.H
S, T = Gate.S, Gate.T # S aka P
SWAP = Gate.SWAP
CX = Gate.CN


def parse(s):
    assert len(s) <= MAX_GATE_SIZE
    ops = [getattr(Gate, c) for c in s]
    op = reduce(matmul, ops)
    return op


class StabilizerCode(object):
    def __init__(self, ops):
        ops = ops.split()
        self.n = len(ops[0])
        ops = [parse(s) for s in ops]
        for g in ops:
          for h in ops:
            assert g*h == h*g
        self.ops = list(ops)

    def get_projector(self):
        G = mulclose(self.ops)
    
        # build projector onto codespace
        P = (1./len(G))*reduce(add, G)
        return P



def test():

    XI = X@I
    ZI = Z@I
    XX = X@X
    ZZ = Z@Z
    IX = I@X
    IZ = I@Z

    #print(CX.shortstr())

    #print()
    #for op in [XI*CX, IX*CX, CX*XI, CX*IX]:
    #    print(op.shortstr())

    assert IX*CX == CX*IX

    #print()
    #for op in [XX*CX, CX*XX]:
    #    print(op.shortstr())

    assert XX*CX == CX*XI
    assert CX*XX == XI*CX

    #print()
    #for op in [ZI*CX, IZ*CX, CX*ZI, CX*IZ, ZZ*CX, CX*ZZ]:
    #    print(op.shortstr())
    assert ZI*CX == CX*ZI
    assert IZ*CX == CX*ZZ
    assert CX*IZ == ZZ*CX

    # five qubit code ----------------

    code = StabilizerCode('XZZXI IXZZX XIXZZ ZXIXZ')
    P = code.get_projector()

    #show_spec(P)

    T = S*H
    A = T@T@T@T@T
    assert A*P == P*A

    for s in 'XXXXX ZZZZZ YYYYY'.split():
        A = parse(s)
        assert A*P == P*A

    A = parse('HHHHH')
    assert A*P != P*A

    A = parse('SSSSS')
    assert A*P != P*A

    # steane code ----------------

    code = StabilizerCode("XXXXIII XXIIXXI XIXIXIX ZZZZIII ZZIIZZI ZIZIZIZ")
    P = code.get_projector()

    for s in 'XZSH':
        s = s*code.n
        A = parse(s)
        assert A*P == P*A

    print("OK")


if __name__ == "__main__":

    name = argv.next() or "test"

    fn = eval(name)
    fn()


