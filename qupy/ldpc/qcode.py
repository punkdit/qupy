#!/usr/bin/env python3


"""
Build symplectic matrices over F_2.
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
from qupy.ldpc.decoder import StarDynamicDistance


def symplectic_form(n):
    A = zeros2(2*n, 2*n)
    I = identity2(n)
    A[:n, n:] = I
    A[n:, :n] = I
    return A


def build_code(stabs):
    m = len(stabs)
    n = len(stabs[0])
    A = zeros2(m, 2*n)
    for i,stab in enumerate(stabs):
        for j,c in enumerate(stab):
            if c=='I':
                pass
            elif c=='X' or c=='Y':
                A[i,j] = 1
            elif c=='Z' or c=='Y':
                A[i,j+n] = 1
    return A



def test_code():
    stabs = "IXXXX XIZZX ZXIZZ XZXIZ ZIZIZ".split()
    A = build_code(stabs)
    m, nn = A.shape
    n = nn//2

    #A = A.transpose()
    F = symplectic_form(n)
    lhs = dot2(A, F, A.transpose())
    assert lhs.sum() == 0

    B = row_reduce(A)
    lhs = dot2(B, F, B.transpose())
    assert lhs.sum() == 0

    W = zeros2(m+1, 2*n)
    W[:m] = A
    for v in cross([(0,1)]*2*n):
        v = array2(v)
        if v.sum()==0:
            continue
        v.shape = (2*n,)
        W[m] = v
        WFW = dot2(W, F, W.transpose())
        for i in range(m):
            w = WFW[m,i]
            if (w==1) != (i==m-1):
                break
        else:
            print(WFW)
            break

    print("W:")
    print(W)
    print("WFW:")
    print(WFW)



if __name__ == "__main__":

    name = argv.next()

    if argv.profile:
        import cProfile as profile
        profile.run("%s()"%name)

    elif name:

        fn = eval(name)
        fn()

    else:

        test_symplectic()
        test_isotropic()

    print("OK\n")



