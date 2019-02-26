#!/usr/bin/env python3

"""
Actually simulate what happens to a state
during a round of error correction.
"""

from random import random
from operator import matmul
from functools import reduce

import numpy
import scipy

from qupy.dense import Qu, Gate
from qupy.util import mulclose, FuzzyDict
from qupy.scalar import EPSILON
from qupy.tool import fstr, write
from qupy.argv import argv
from qupy.ldpc import solve

I = Gate.I
X = Gate.X
Z = Gate.Z
Y = Gate.Y


def sumops(ops):
    op = None
    for opi in ops:
        if op is None:
            op = opi
        else:
            op = op + opi
    return op

"""
The Lie algebra of U(n) consists of n × n skew-Hermitian matrices.
"""


def test_errop(t):
    H = Gate.random_hermitian(2)
    A = -1.j*t*H
    print(A)
    for val, vec in A.eigs():
        print(val, numpy.exp(val))
    print()
    U = A.expm()
    for val, vec in U.eigs():
        print(val)
    print()
    assert U.is_unitary()
    return U


def errop(t):

    H = Gate.random_hermitian(2)
    U = H.evolution_operator(t)
    assert U.is_unitary()

    return U


def measure(H, v):
    assert H.is_hermitian()
    d = FuzzyDict()
    for val, vec in H.eigs():
        assert abs(val.imag) < EPSILON
        val = val.real
        space = d.get(val, [])
        space.append(vec)
        d[val] = space
    povm = []
    assert abs(1. - v.norm()) < EPSILON
    x = random()
    total = 0.
    result = None
    value = None
    ps = []
    for val, space in d.items():
        op = Qu(H.shape, H.valence)
        for vec in space:
            op += vec @ ~vec
        assert op*op == op
        u = op*v
        r = ~u * v
        #print(fstr(r))
        total += r
        if result is None and x < total:
            result = u / u.norm()
            value = val
        ps.append(r)
    #print(fstr(sum(ps)))
    return value, result


#class Code(object):
#    def __init__(self, H, L):


def sy_form(n):
    J = solve.zeros2((2*n, 2*n))
    for i in range(n):
        J[2*i+1, 2*i] = 1
        J[2*i, 2*i+1] = 1
    return J


#def str2op(s):


def build(items):
    items = items.strip().split()
    #print(items)
    rows = []
    n = None
    for item in items:
        assert n is None or len(items)==n
        n = len(items)
        row = []
        for i in item:
            if i=="X":
                row += [1, 0]
            elif i=="Z":
                row += [0, 1]
            elif i=="Y":
                row += [1, 1]
            elif i=="I":
                row += [0, 0]
            else:
                assert 0
        rows.append(row)
    H = solve.array2(rows)
    #print(H)
    J = sy_form(n)
    HJ = solve.dot2(H, J)
    #print(HJ)
    T = solve.pseudo_inverse(HJ.transpose())
    #print(T)
    TJ = solve.dot2(T, J)
    #print(TJ)
    #print(solve.dot2(TJ, H.transpose())) # identity

    ops = []
    for row in T:
        op = []
        for i in range(n):
            opi = list(row[2*i : 2*(i+1)])
            if opi == [0, 0]:
                op.append(I)
                #write("I")
            elif opi == [1, 0]:
                op.append(X)
                #write("X")
            elif opi == [0, 1]:
                op.append(Z)
                #write("Z")
            elif opi == [1, 1]:
                op.append(Y)
                #write("Y")
            else:
                assert 0, opi
        op = reduce(matmul, op)
        ops.append(op)
        #write("\n")
    return ops
            

def main():

    t = argv.get("t", 0.02)

    T = build("""
    XZZXI 
    IXZZX
    XIXZZ 
    ZXIXZ
    XXXXX
    """)

    n = 5
    stabs = [ 
        X@Z@Z@X@I, 
        I@X@Z@Z@X, 
        X@I@X@Z@Z, 
        Z@X@I@X@Z]
    Lx = X@X@X@X@X
    #Lz = Z@Z@Z@Z@Z
    Lz = T[n-1]

    ops = stabs + [Lx]
    for i in range(n):
      for j in range(n):
        c = (T[i]*ops[j] == ops[j]*T[i])
        assert c == (i!=j)
        a = (T[i]*ops[j] == -ops[j]*T[i])
        assert a == (i==j)

    G = mulclose(stabs)
    assert len(G) == 16
    N = len(G)
    P = (1./N) * sumops(G)

    assert P*P == P

    for op in stabs:
        assert Lx*op == op*Lx
        assert Lz*op == op*Lz

    trials = argv.get("trials", 100)

    for trial in range(trials):

        if 1:
            # noise on one qubit
            U = errop(t)
            U = U@I@I@I@I
        else:
            # noise on all qubits
            Us = []
            for i in range(n):
                Us.append(errop(t))
            U = reduce(matmul, Us)

        v = Qu.random((2,)*n, "u"*n)
        v = P*v
        v.normalize()
        #print(v.flatstr())

        u = U*v
        u.normalize()
    
        for i, op in enumerate(stabs):
            val, u = measure(op, u)
            print(fstr(val), end=" ")
            if val < 0:
                #print("*")
                u = T[i]*u
        #print()
    
        #for i, op in enumerate(stabs):
            #val, v = measure(op, v)
            #print(fstr(val), end=" ")
        #print()
    
        assert P*u == u
        assert P*v == v

        phase = None
        for uu in [u, Lx*u, Lz*u, Lx*Lz*u]:
            r = ~uu * v
            if abs(1. - abs(r)) < EPSILON:
                #write("+")
                phase = r
            #else:
                #write(".")
        #write("\n")
        if phase is not None:
            print(phase)
        else:
            print()

        #print(v==u, v==Lx*u, v==Lz*u, v==Lx*Lz*u)
        #print(v==-u, v==-Lx*u, v==-Lz*u, v==-Lx*Lz*u)

        #print(v.flatstr())
        #print((Lz*u).flatstr())
        #print(~u * v)
        


if __name__ == "__main__":

    main()

