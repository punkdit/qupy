#!/usr/bin/env python3

"""
See:
Quantum Teleportation is a Universal Computational Primitive
Daniel Gottesman, Isaac L. Chuang
https://arxiv.org/abs/quant-ph/9908010
"""

from math import sqrt
from random import choice, randint, seed, shuffle
from functools import reduce
from operator import mul, matmul, add

import numpy

from qupy import scalar
#scalar.use_reals()

from qupy.scalar import EPSILON, MAX_GATE_SIZE

from qupy.abstract import Space
from qupy.dense import Qu, Gate, Vector, bitvec
from qupy.tool import cross
from qupy.argv import argv


I, X, Z, H = Gate.I, Gate.X, Gate.Z, Gate.H
S, T = Gate.S, Gate.T # S aka P
SWAP = Gate.SWAP
CX = Gate.CN
CZ = Z.control()

ir2 = 1./sqrt(2)
zero = bitvec(0)
one = bitvec(1)
plus = ir2 * (bitvec(0) + bitvec(1))
minus = ir2 * (bitvec(0) - bitvec(1))
bell = ir2 * (bitvec(0, 0) + bitvec(1, 1))


def test():
    assert CX == X.control()

    assert CX == X.control(1, 0) # target, source
    assert CX != X.control(0, 1) # target, source

    assert CZ == Z.control(1, 0) # target, source
    assert CZ == Z.control(0, 1) # target, source

    #assert CZ * (H @ H) * CZ == SWAP

    assert (I@H) * CZ * (I@H) == CX 
    assert (I@H) * CX * (I@H) == CZ
    assert CX * (I@H) == (I@H) * CZ
    assert CZ * (I@H) == (I@H) * CX

    assert Z.control(2, 4, rank=6) == I @ I @ Z.control(0, 2) @ I 

    assert H.bounce(2, rank=6) == I@I@H@I@I@I


def state_teleport():

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


def gate_teleport():

    CX = X.control(0, 1) # target, source

    chi = (I @ CX @ I) * (bell @ bell)
    
    B = (H @ H) * CZ

    lhs = (B @ I @ I @ B) * (I @ chi @ I)

    #print(lhs.shortstr())

    lhs = X.control(2, 4, rank=6) * lhs 
    #lhs = Z.control(3, 4, rank=6) * lhs 
    lhs = X.control(2, 5, rank=6) * lhs 

    lhs = Z.control(2, 0, rank=6) * lhs 
    lhs = Z.control(3, 0, rank=6) * lhs 
    lhs = Z.control(3, 1, rank=6) * lhs 

    x = plus
    lhs = (~x @ ~x @ I @ I @ ~x @ ~x) * lhs
    print(lhs.valence)
    print(lhs.shortstr())

    print(CX.valence)
    print(CX.shortstr())



def search():

    CX = X.control(0, 1) # target, source

    assert CZ == Z.control(0, 1)
    assert CZ == Z.control(1, 0)
    #print(CZ.flat().v)

    #print(X.control(0, 1).flat().v)
    #print(X.control(1, 0).flat().v)

    #chi = (I @ CX @ I) * (bell @ bell) # looking to teleport a CX gate
    #B = (H @ H) * CZ

    chi = (I @ CZ @ I) * (bell @ bell) # looking to teleport a CZ gate
    B = CZ

    start = (B @ I @ I @ B) * (I @ chi @ I)

    for idxs in cross([(0, 1, 2)]*6):

        ops = [(I, X, Z)[i] for i in idxs]

        # IXX XIX
    
        lhs = ops[0].control(2, 4, rank=6) * start 
        lhs = ops[1].control(3, 4, rank=6) * lhs 
        lhs = ops[2].control(2, 5, rank=6) * lhs 
    
        lhs = ops[3].control(2, 0, rank=6) * lhs 
        lhs = ops[4].control(3, 0, rank=6) * lhs 
        lhs = ops[5].control(3, 1, rank=6) * lhs 

        lhs0 = lhs

        x = plus
        #x = one
        lhs = (~x @ ~x @ I @ I @ ~x @ ~x) * lhs
        #print(lhs.valence)
        #print(lhs.valence, lhs.shortstr(), lhs[0,0,0,0])
        r = lhs[0, 0, 0, 0]
    
        if abs(r) < EPSILON:
            continue

        lhs /= r

        if lhs == CZ:
            print("FOUND "*10)
            print(lhs.shortstr())
            print(['IXZ'[i] for i in idxs])

            #break


    if 0:
        print(lhs0.shortstr())
        op = (Z.control(0, 5) * (I @ plus @ plus @ plus @ plus @ I))
        print(op.shortstr())
    


    #print(lhs.valence)
    #print(lhs.shortstr())
    #print(CX.valence)
    #print(CX.shortstr())


def main():

    cap = ~bitvec(0, 0) + ~bitvec(1, 1)
    cup = bitvec(0, 0) + bitvec(1, 1)


    lhs = (cap @ I) * (I @ cup)
    print(lhs)


if __name__ == "__main__":

    test()
    state_teleport()
    #gate_teleport()
    #search()

    main()



