#!/usr/bin/env python3

from math import sqrt, pi
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



def test_1():

    assert S*S == Z

    i = 1.j
    phase = numpy.exp(i*pi/8)
    C1 = mulclose([X, Z, phase*I]) # Pauli group
    C2 = mulclose([H, S, phase*I]) # Clifford group

    assert len(C1) == 64, len(C1)
    assert len(C2) == 384, len(C2)

    r = numpy.exp(-pi*i/8)/sqrt(2)
    v = r*numpy.array([[1., i], [i, 1.]])
    M = Qu((2, 2), 'ud', v)

    for g in C2:
        ginv = inv(g)
        assert g*ginv == I

    for g2 in C2:
        g2inv = inv(g2)
        assert g2*g2inv == I
        for g in C1:
            k = g2 * g * g2inv
            assert find(k, C1)

    print(M)
    print(find(M, C2))
    for g in C1:
        k = M * g * inv(M)
        assert find(k, C1)


def test_2():
    "two qubits"

    i = 1.j
    phase = numpy.exp(i*pi/8)

    II = I@I
    XI = X@I
    IX = I@X
    ZI = Z@I
    IZ = I@Z
    gen = [XI, IX, ZI, IZ, phase*II]
    gen = [op.flat() for op in gen]
    C1 = mulclose(gen)

    assert len(C1) == 256, len(C1)

#    CZ = Z.control()
#    op = CZ.flat()
#    print(op)
#    print(op.valence)

    r = numpy.exp(-pi*i/8)
    v = r*numpy.array([
        [1., 0., 0., 0.],
        [0., 1.j, 0., 0.],
        [0., 0., 1.j, 0.],
        [0., 0., 0., 1.]])
    CZ = Qu((4, 4), 'ud', v)

    II = II.flat()
    assert (CZ*~CZ) == II

    assert not find(CZ, C1)
    for g in C1:
        k = CZ * g * ~CZ
        assert find(k, C1)

    r = numpy.exp(pi*i/8)/sqrt(2)
    v = r*numpy.array([[1., i], [i, 1.]])
    H = Qu((2, 2), 'ud', v)

    plus = 1./sqrt(2) * (bitvec(0) + bitvec(1)) 
    #print(plus)
    #print(H*plus) # proportional to plus state

    IH = (I@H).flat()
    HI = (H@I).flat()
    HH = (H@H).flat()

    assert IH * CZ * IH == CZ * IH * CZ # Reidemeister III move

    g = CZ * HH * CZ
    assert g*g == II
    assert g != SWAP.flat()

    A = (CZ @ I).flat()
    B = (I @ CZ).flat()
    III = (II@I).flat()

    assert A*B == B*A

    ABA = A*B*A
    BAB = B*A*B
    r = numpy.exp(pi*i/8)
    print((ABA/r).shortstr())
    print((BAB/r).shortstr())
    #assert A*B*A == B*A*B
    op = III
    for i in range(1, 100):
        op = ABA * op
        if op == III:
            break
    else:
        assert 0
    print(i)

    assert ABA ** 16 == III
    assert BAB ** 16 == III

if __name__ == "__main__":

    name = argv.next() or "test"

    fn = eval(name)
    fn()


