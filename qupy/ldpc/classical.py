#!/usr/bin/env python3

"""
Part of the calculation in homological.py

"""

import sys
from random import randint, seed, random, shuffle, choice

import numpy
import numpy.random as ra
from numpy.linalg import lstsq

from qupy.argv import argv
from qupy.tool import write, choose
from qupy.ldpc.css import CSSCode, randcss
from qupy.ldpc.solve import shortstr, zeros2, array2, dot2, parse, identity2, rank
from qupy.ldpc.solve import find_kernel, cokernel, eq2, get_reductor, intersect
from qupy.ldpc.solve import linear_independent, solve, row_reduce, rand2
from qupy.ldpc.solve import remove_dependent, dependent_rows, span
from qupy.ldpc.gallagher import make_gallagher, classical_distance



def in_support(H, keep_idxs, check=False):
    # find span of H contained within idxs support
    n = H.shape[1]
    remove_idxs = [i for i in range(n) if i not in keep_idxs]
    A = identity2(n)
    A = A[keep_idxs]
    H1 = intersect(A, H)

    if check:
        lhs = set(str(x) for x in span(A))
        rhs = set(str(x) for x in span(H))
        meet = lhs.intersection(rhs)
        assert meet == set(str(x) for x in span(H1))
    
    return H1


def min_weight(G):
    k, n = G.shape
    dist = n
    Gt = G.transpose()
    for u in numpy.ndindex((2,)*G.shape[0]):
        v = dot2(Gt, u)
        if 0 < v.sum() < dist:
            dist = v.sum()
    return dist



def test(n, k, verbose=False):
    assert n > k

    while 1:
        G = rand2(k, n)
        if rank(G) < k:
            continue
        dist = min_weight(G)
        if dist > 1:
            break

    if verbose:
        print(shortstr(G))
        print("dist =", dist)
        print()

    while 1:
        idxs = set()
        while len(idxs) < k:
            idx = randint(0, n-1)
            idxs.add(idx)
        idxs = list(idxs)
        idxs.sort()

        G1 = in_support(G, idxs)
        if len(G1):
            continue

        jdxs = set()
        while len(jdxs) < k:
            jdx = randint(0, n-1)
            if jdx not in idxs:
                jdxs.add(jdx)
        jdxs = list(jdxs)
        jdxs.sort()

        G2 = in_support(G, jdxs)
        if len(G2) == 0:
            break


    if verbose:
        v = zeros2(1, n)
        v[:, idxs] = 1
        print(shortstr(v))
    
        v = zeros2(1, n)
        v[:, jdxs] = 1
        print(shortstr(v))


def main():

    n = argv.get("n", 10)
    k = argv.get("k", 5)

    while 1:
        test(n, k)
        c = choice("/\\")
        print(c, flush=True, end="")



if __name__ == "__main__":

    _seed = argv.get("seed")
    if _seed is not None:
        seed(_seed)
        ra.seed(_seed)

    main()


