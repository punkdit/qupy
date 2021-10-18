#!/usr/bin/env python3

"""
Represent Clifford's as affine symplectic transforms over Z/2 .
This is the qubit clifford group modulo phases.

"""

from collections import namedtuple
from functools import reduce
from operator import mul
from random import shuffle

import numpy
from numpy import concatenate as cat

from qupy.util import mulclose_fast
from qupy.tool import cross, choose
from qupy.argv import argv
from qupy.smap import SMap
from qupy.ldpc.solve import zeros2, shortstr, solve, dot2, array2, eq2, parse, pseudo_inverse, identity2
from qupy.ldpc.solve import enum2, row_reduce
from qupy.ldpc.css import CSSCode


class Clifford(object):
    """
    """

    def __init__(self, A):
        assert A.data.c_contiguous, "need to copy A first"
    
        m, n = A.shape
        assert m == n
        assert n%2 == 1
        self.shape = (n, n)
        self.n = n
        assert A[n-1, n-1] == 1, A
        assert numpy.abs(A[n-1]).sum() == 1, A

        self.A = A
        self.key = A.tobytes() # needs A to be c_contiguous 

    def __str__(self):
        #s = str(self.A)
        #s = s.replace("0", ".")
        s = shortstr(self.A)
        return s

    def __mul__(self, other):
        assert isinstance(other, Clifford)
        assert other.shape == self.shape
        A = dot2(self.A, other.A)
        return Clifford(A)

    def __getitem__(self, key):
        return self.A[key]

    def __setitem__(self, key, value):
        assert 0, "do we really need this?"
        self.A[key] = value
        self.key = A.tobytes()

    def transpose(self):
        A = self.A.transpose().copy()
        return Clifford(A)

    def inverse(self):
        A = pseudo_inverse(self.A)
        return Clifford(A)

    def __eq__(self, other):
        assert isinstance(other, Clifford)
        assert self.shape == other.shape
        #return eq2(self.A, other.A)
        return self.key == other.key

    def __ne__(self, other):
        assert isinstance(other, Clifford)
        assert self.shape == other.shape
        return self.key != other.key

    def __hash__(self):
        # warning: i am mutable
        return hash(self.key)

    @classmethod
    def identity(cls, n):
        A = zeros2(2*n+1, 2*n+1)
        for i in range(2*n+1):
            A[i, i] = 1
        return Clifford(A)

    @classmethod
    def z(cls, n, idx):
        A = zeros2(2*n+1, 2*n+1)
        for i in range(2*n+1):
            A[i, i] = 1
        A[idx, 2*n] = 1
        return Clifford(A)

    @classmethod
    def x(cls, n, idx):
        A = zeros2(2*n+1, 2*n+1)
        for i in range(2*n+1):
            A[i, i] = 1
        A[n+idx, 2*n] = 1
        return Clifford(A)

    @classmethod
    def hadamard(cls, n, idx):
        A = zeros2(2*n+1, 2*n+1)
        for i in range(2*n):
            if i==idx:
                A[i, n+i] = 1
            elif i==n+idx:
                A[i, i-n] = 1
            else:
                A[i, i] = 1
        A[2*n, 2*n] = 1
        return Clifford(A)

    @classmethod
    def cnot(cls, n, src, tgt):
        assert src!=tgt
        A = zeros2(2*n+1, 2*n+1)
        for i in range(2*n+1):
            A[i, i] = 1
        A[2*n, 2*n] = 1
        A[src, tgt] = 1
        A[tgt+n, src+n] = 1
        return Clifford(A)

    @classmethod
    def cz(cls, n, src, tgt):
        CN = cls.cnot(n, src, tgt)
        H = cls.hadamard(n, tgt)
        CZ = H * CN * H
        return CZ

    @classmethod
    def s(cls, n, i):
        A = cls.identity(n).A
        assert 0<=i<n
        #A[i+n, i] = 1
        A[i, i+n] = 1
        A[i+n, 2*n] = 1
        return Clifford(A)
    
    @classmethod
    def swap(cls, n, idx, jdx):
        A = zeros2(2*n+1, 2*n+1)
        A[2*n, 2*n] = 1
        assert idx != jdx
        for i in range(n):
            if i==idx:
                A[i, jdx] = 1 # X sector
                A[i+n, jdx+n] = 1 # Z sector
            elif i==jdx:
                A[i, idx] = 1 # X sector
                A[i+n, idx+n] = 1 # Z sector
            else:
                A[i, i] = 1 # X sector
                A[i+n, i+n] = 1 # Z sector
        return Clifford(A)


def test():

    # --------------------------------------------
    # Clifford group order is 24

    n = 1
    I = Clifford.identity(n)

    X = Clifford.x(n, 0)
    #print(X)
    #print()
    Z = Clifford.z(n, 0)
    #print(Z)
    #print()

    S = Clifford.s(n, 0)
    #print(S)
    #print()

    H = Clifford.hadamard(n, 0)

    assert X != I
    assert X*X == I

    assert Z*X == X*Z # looses the phase 
    assert H*H == I
    assert H*X*H == Z
    assert H*Z*H == X

    assert S*S == Z

    assert S*Z == Z*S
    assert S*X != X*S
    assert S*S*S*S == I

    assert S*H*S*H*S*H == I

    G = mulclose_fast([S, H])
    assert len(G) == 24

    # --------------------------------------------
    # Clifford group order is 11520

    n = 2
    II = Clifford.identity(n)

    XI = Clifford.x(n, 0)
    IX = Clifford.x(n, 1)
    XX = XI * IX
    ZI = Clifford.z(n, 0)
    IZ = Clifford.z(n, 1)
    ZZ = ZI * IZ
    SI = Clifford.s(n, 0)
    IS = Clifford.s(n, 1)
    SS = SI * IS
    HI = Clifford.hadamard(n, 0)
    IH = Clifford.hadamard(n, 1)

    CX = Clifford.cnot(n, 0, 1)
    CX1 = Clifford.cnot(n, 1, 0)
    CZ = Clifford.cz(n, 0, 1)
    CZ1 = Clifford.cz(n, 1, 0)

    assert SI*SI == ZI

    assert SI*ZI == ZI*SI
    assert SI*XI != XI*SI
    assert SI*SI*SI*SI == II

    assert CX * CX == II
    assert CZ * CZ == II
    assert CZ1 == CZ

    assert CX * IX == IX * CX
    assert CX * XI * CX == XX

    assert CX * ZI == ZI * CX
    assert CX * IZ * CX == ZZ

    SWAP = Clifford.swap(n, 0, 1)
    assert SWAP * ZI == IZ * SWAP
    assert SWAP * XI == IX * SWAP

    assert CX * CX1 * CX == SWAP

    assert CZ == IH * CX * IH

    assert CZ * ZI == ZI * CZ
    assert CZ * IZ == IZ * CZ

    assert CZ * XI * CZ == XI*IZ
    assert CZ * IX * CZ == IX*ZI

    #print(CX*CX1)
    #print()
    #print(SWAP)

    G = mulclose_fast([SI, IS, CX, HI, IH ])
    assert len(G) == 11520

    # --------------------------------------------

    n = 5
    I = Clifford.identity(n)
    CZ = Clifford.cz(n, 0, 1)
    SWAP = Clifford.swap(n, 0, 1)
    assert CZ*CZ == I
    assert SWAP*CZ == CZ*SWAP



if __name__ == "__main__":

    name = argv.next()

    if argv.profile:
        import cProfile as profile
        profile.run("%s()"%name)

    elif name:

        fn = eval(name)
        fn()

    else:

        test()

    print("OK")



