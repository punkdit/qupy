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
from qupy.ldpc.clifford import all_codes



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


def min_weight(G, max_dist=0):
    k, n = G.shape
    dist = n
    Gt = G.transpose()
    for u in numpy.ndindex((2,)*G.shape[0]):
        v = dot2(Gt, u)
        if 0 < v.sum() < dist:
            dist = v.sum()
        #if dist<=max_dist:
        #    break
    return dist



def search(G, H, max_tries=None):

    m, n = H.shape
    k, n1 = G.shape
    assert n==n1
    assert m+k==n

    count = 0
    while 1:
        count += 1
        if max_tries is not None and count>max_tries:
            return False

        idxs = set()
        while len(idxs) < k:
            idx = randint(0, n-1)
            idxs.add(idx)
        idxs = list(idxs)
        idxs.sort()

        if len(in_support(G, idxs)):
            #print("1", end="", flush=True)
            continue

        if len(in_support(H, idxs)):
            #print("2", end="", flush=True)
            continue

        jdxs = set()
        while len(jdxs) < k:
            jdx = randint(0, n-1)
            if jdx not in idxs:
                jdxs.add(jdx)
        jdxs = list(jdxs)
        jdxs.sort()

        if len(in_support(G, jdxs)):
            #print("3", end="", flush=True)
            continue

        if len(in_support(H, jdxs)):
            #print("4", end="", flush=True)
            continue

        break

    #print()

    if argv.verbose:
        v = zeros2(1, n)
        v[:, idxs] = 1
        print(shortstr(v))
    
        v = zeros2(1, n)
        v[:, jdxs] = 1
        print(shortstr(v))

    return True


def test(n, k, dist=2, verbose=False):
    assert n > k

    if argv.rand:
        while 1:
            G = rand2(k, n)
            if rank(G) < k:
                continue
            dG = min_weight(G)
            if dG < dist:
                continue

            H = find_kernel(G)
            dH = min_weight(H)
            if dH < dist:
                continue

            break

    else:
        G = zeros2(k, n)
        jdx = 0
        for idx in range(k):
          for kdx in range(dist):
            G[idx,jdx+kdx] = 1
          jdx += dist-1

        dG = min_weight(G) if n < 20 else None
        assert dG is None or dG == dist

        H = find_kernel(G)

    #print(".", flush=True, end="")
    H = row_reduce(H)

    search(G, H)

    if verbose:
        print("G =")
        print(shortstr(G))
        print("weight =", dG)
        print()

        print("H =")
        print(shortstr(H))
        print()


def main():

    n = argv.get("n", 10)
    k = argv.get("k", 4)
    dist = argv.get("dist", 2)
    max_tries = argv.get("max_tries", 1000)
    verbose = argv.verbose

    count = 0
    fails = 0

    for G in all_codes(k, n):
        assert rank(G) == k
        dG = min_weight(G, dist)
        if dG < dist:
            #print(".", end="", flush=True)
            continue

        H = find_kernel(G)
        dH = min_weight(H, dist)
        if dH < dist:
            print("*", end="", flush=True)
            continue

        print("G =")
        print(shortstr(G))
        print("H =")
        print(shortstr(H))
        result = search(G, H, max_tries)
        count += 1
        if result:
            print("\n")
        else:
            print("XXXXXXXXXXXXXXXXXXX FAIL\n")
            fails += 1

    print("codes found: %d, fails %d"%(count, fails))

#
#    while 1:
#        test(n, k, dist, verbose)
#
#        if not verbose:
#            c = choice("/\\")
#            print(c, flush=True, end="")
#
#        if not argv.forever:
#            break



if __name__ == "__main__":

    _seed = argv.get("seed")
    if _seed is not None:
        seed(_seed)
        ra.seed(_seed)

    main()


