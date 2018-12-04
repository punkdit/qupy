#!/usr/bin/env python3

from math import sqrt
from random import choice, randint, seed, shuffle
from functools import reduce
from operator import mul, matmul, add

import numpy

from qupy import scalar
#scalar.use_reals()

from qupy.scalar import EPSILON

from qupy.abstract import Space
from qupy.dense import Qu, Gate, Vector
#from qupy.dense import genidx, bits, is_close, on, off, scalar
from qupy.dense import bitvec
from qupy.argv import argv
from qupy.util import mulclose, show_spec, FuzzyDict


I, X, Z, H = Gate.I, Gate.X, Gate.Z, Gate.H
S, T = Gate.S, Gate.T
SWAP = Gate.SWAP


class POVM(object):
    def __init__(self, items):
        "items: list of pairs (proj, value)"
        self.items = items

    @classmethod
    def fromop(cls, A, epsilon=EPSILON):
        projs = []
        items = FuzzyDict(EPSILON)
        for val, vec in swap.eigs():
            P1 = vec @ ~vec
            P = items.get(val)
            P = P1 if P is None else P+P1
            items[val] = P
        items = list(items.items())
        return cls(items)



def test_universal():

    G = mulclose([Z, X]) # real pauli group
    assert len(G) == 8

    G = mulclose([Z, X, S])
    assert len(G) == 32

    G = mulclose([Z, X, S, T])
    assert len(G) == 128

    G = mulclose([Z, X, S, H])
    assert len(G) == 192

    G = mulclose([Z, X, S, T, H], maxsize=1000) # infinite group
    assert len(G) >= 1000


def test():
    #global H, X, Z, I

    assert (H*Z) == (X*H)

    # build transversal gates
    TI = I@I@I
    TH = H@H@H
    TX = X@X@X
    TZ = Z@Z@Z

    assert (TH*TZ) == (TX*TH)

    # The S gate
    TS = S@S@S

    # The T gate
    TT = T@T@T

    # the 4-dim irrep of GL(2)
    irrep4 = [
        bitvec(0, 0, 0),
        (bitvec(1, 0, 0) + bitvec(0, 1, 0) + bitvec(0, 0, 1)).normalized(),
        (bitvec(0, 1, 1) + bitvec(1, 0, 1) + bitvec(1, 1, 0)).normalized(),
        bitvec(1, 1, 1)]

    # encode into irrep4:
    B = reduce(add, [ v @ ~bitvec(i, base=len(irrep4)) for (i, v) in enumerate(irrep4)])

    # the 2-dim irrep of GL(2)
    irrep2 = bitvec(0, 1, 0) - bitvec(1, 0, 0)

    #Encode = bitvec(0, 0, 0) @ ~bitvec(0) + bitvec(1, 1, 1) @ ~bitvec(1)
    #Decode = bitvec(0) @ ~bitvec(0, 0, 0) + bitvec(1) @ ~bitvec(1, 1, 1)

    v0, v1, v2, v3 = irrep4
    e0 = (v0+v2).normalized()
    e1 = (v1+v3).normalized()

    e0 = v0
    e1 = v3

    Encode = e0 @ ~bitvec(0) + e1 @ ~bitvec(1)
    Decode = bitvec(0) @ ~e0 + bitvec(1) @ ~e1

    assert (Decode * Encode) == I

    if 0:
        error = Z@I@I
        lhs = Decode * error * Encode
        rhs = I
        print("error:")
        print(lhs.shortstr())
        # ???
        return

    for A in [X, Z, H]:
        TA = A@A@A
        lhs = Decode * TA * Encode 
        rhs = A
        #print(lhs.shortstr())
        #print(rhs.shortstr())
        #print(lhs == rhs)

    # ----------------------------------------------

    gen = [TH, TX, TZ]

    G = mulclose(gen) # generate a group
    assert len(G) == 16

    # make 4 dimensional rep
    GB = [(~B) * g * B for g in G]

    # is it irreducible ? No
    chi = [complex(g.trace()) for g in GB]
    x = sum((x*x.conjugate()).real for x in chi)/len(G)
    assert abs(x-2.) < EPSILON

    # ----------------------------------------------

    gen = [TH, TX, TZ, TS] # adding TT generates infinite group
    # make 4 dimensional rep
    gen = [(~B) * g * B for g in gen]

    G = mulclose(gen) # generate a group
    assert len(G) == 192

    # is it irreducible ? Yes
    chi = [complex(g.trace()) for g in G]
    x = sum((x*x.conjugate()).real for x in chi)/len(G)
    assert abs(x-1.) < EPSILON

    # ----------------------------------------------

    print("OK")
    


def main():

    n = argv.get("n", 3)

    III = I@I@I
    shape, valence = III.shape, III.valence

    HHH = H@H@H

    s0 = SWAP@I
    s1 = I@SWAP

    assert (s0*s0).is_close(III)
    assert (s1*s1).is_close(III)
    assert (s0*s1*s0).is_close(s1*s0*s1)

    swaps = [s0, s1]

    P = (1./6) * (III + s0 + s1 + s0*s1 + s1*s0 + s0*s1*s0)
    #show_spec(P)

    v = Qu((2,)*n, 'u'*n)
    v[1, 1, 1] = 1.
    print(v.shortstr())
    print((HHH * v).shortstr())

    return

    Ps = []
    for swap in swaps:
        P = Qu(shape, valence)
        for val, vec in swap.eigs():
            print(val, vec.shortstr())
            if val > 0:
                P = P + val*(vec @ ~vec)

        Ps.append(P)
        print()

        assert (P*P).is_close(P)

    P, Q = Ps
    #assert (P*Q*P).is_close(Q*P*Q)
    P = (P*Q + Q*P)
    show_spec(P)
    assert (P*P).is_close(P)
    
    #print("dim:", len([val for val, vec in items if val>0]))
    


if __name__ == "__main__":

    name = argv.next() or "test"

    fn = eval(name)
    fn()


