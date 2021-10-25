#!/usr/bin/env python3

"""
Represent Clifford's as affine symplectic _transforms over Z/2 .
This is the qubit clifford group modulo phases.

possible todo: implement phases as a group extension:
    phases >---> Clifford --->> AffineSymplectic

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


def order(n):
    # size of Clifford group, modulo phases
    sz = 1
    for i in range(1, n+1):
        sz *= 2 * (4**1 - 1) * 4**i
    return sz

_cache = {}
def symplectic_form(n):
    F = _cache.get(n)
    if F is None:
        F = zeros2(2*n, 2*n)
        for i in range(n):
            F[n+i, i] = 1
            F[i, n+i] = 1
        _cache[n] = F
    F = F.copy()
    return F


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
        #assert self.check()

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

#    def inverse(self):
#        A = pseudo_inverse(self.A)
#        return Clifford(A)

    def inverse(self):
        A = self.A
        nn = self.n
        n = (nn-1)//2
        B = A[:2*n, :2*n] # symplectic 
        v = A[:2*n, 2*n]  # translation, shape (2*n,)
        F = symplectic_form(n)
        Fi = F # an involution
        Bi = dot2(Fi, dot2(B.transpose()), F)
        A1 = A.copy()
        A1[:2*n, :2*n] = Bi
        A1[:2*n, 2*n] = dot2(-Bi, v)
        return Clifford(A1)

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

    def check(self):
        # check symplectic condition
        A = self.A
        nn = self.n
        n = (nn-1)//2
        B = A[:2*n, :2*n]
        F = symplectic_form(n)
        lhs = dot2(dot2(B.transpose(), F), B)
        return numpy.alltrue(lhs == F)

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

    @classmethod
    def make_op(cls, v, method):
        assert method in [cls.x, cls.z, cls.s, cls.hadamard]
        n = len(v)
        g = Clifford.identity(n)
        for i in range(n):
            if v[i]:
                g *= method(n, i)
        return g

    def get_symplectic(self):
        n = self.n
        A = self.A
        B = A[:n-1, :n-1]
        return B

    def get_translation(self):
        n = self.n
        A = self.A
        v = A[:n-1, n-1]
        return v

    def is_translation(self):
        n = self.n
        A = self.A
        A = A[:n-1, :n-1]
        return numpy.alltrue(A == identity2(n-1))

    def find_encoding(A, Lz, Lx):
        assert len(Lx) == len(Lz)
        Ai = A.inverse()
        rows = []
        for l in Lz + Lx:
            B = A*l*Ai
            assert B.is_translation()
            v = B.get_translation()
            rows.append(v)
        B = numpy.array(rows)
        #B = B[:8, :]
        #print(B.shape)
        #print(shortstr(B))
        U = numpy.array([l.get_translation() for l in Lz+Lx])
        #print(U.shape)
        #print(shortstr(U))
        Ut = U.transpose()
        V = solve(Ut, B.transpose())
        if V is None:
            return None
        print(A.get_translation())
        u = solve(Ut, A.get_translation())
        print( u is not None )
        #print(V.shape)
        #print(shortstr(V))


def test():

    n = 3
    F = symplectic_form(n)
    I = identity2(2*n)
    assert numpy.alltrue(I == dot2(F, F))

    # --------------------------------------------
    # Clifford group order is 24

    n = 1
    I = Clifford.identity(n)

    X = Clifford.x(n, 0)
    Z = Clifford.z(n, 0)
    S = Clifford.s(n, 0)
    Si = S.inverse()
    H = Clifford.hadamard(n, 0)
    Y = X*Z

    print("X =")
    print(X)
    print("Z =")
    print(Z)
    print("S =")
    print(S)
    print("Si =")
    print(Si)
    print()

    assert X != I
    assert Z != I
    assert X*X == I
    assert Z*Z == I
    assert Z*X == X*Z # looses the phase 

    assert Si*S == S*Si == I
    assert Si*Si == Z
    assert S*Z == Z*S
    assert S*X == Y*S
    assert S*Y*Si == X
    assert S*S == Z
    assert S*X != X*S
    assert S*S*S*S == I
    assert S*S*S == Si

#    B = array2([[1, 0], [0, 1], [1, 1]])
#    C = array2([[1, 1], [0, 1], [1, 1]])
#    for bits in cross( [(0,1)]*6 ):
#        A = zeros2(2*n+1, 2*n+1)
#        A[2*n, 2*n] = 1
#        for idx, key in enumerate(numpy.ndindex((2,3))):
#            #print(key, end=" ")
#            A[key] = bits[idx]
#        val = (A[0,0]*A[1,1] - A[0,1]*A[1,0])%2
#        if val != 1:
#            continue
#        op = Clifford(A)
#        #if op*op == Z:
#        #    print(op, '\n')
#        lhs = dot2(A, B)
#        rhs = C
#        if numpy.alltrue(lhs == rhs):
#            print("op =")
#            print(op, "\n")
#            print(op*op, "\n")
#            print(op*op*op, "\n")
#            print(op*op*op*op, "\n")

    assert H*H == I
    assert H*X*H == Z
    assert H*Z*H == X

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

    print("CZ =")
    print(CZ)

    print("CX =")
    print(CX)

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

    for g in G:
        assert g.check()
        h = g.inverse()
        assert h.check()
        assert g*h == II

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



