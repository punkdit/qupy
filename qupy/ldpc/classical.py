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



def search(G, H, max_tries=None, debug=False):

    m, n = H.shape
    k, n1 = G.shape
    assert n==n1
    assert m+k==n

    check = shortstr(H)+shortstr(G)

    count = 0
    while 1:
        assert check == shortstr(H)+shortstr(G)
        count += 1
        if max_tries is not None and count>max_tries:
            print("/\n")
            return False

        idxs = set()
        while len(idxs) < k:
            idx = randint(0, n-1)
            idxs.add(idx)
        idxs = list(idxs)
        idxs.sort()

        #print(in_support(G, idxs))
        #print(in_support(H, idxs))

        if len(in_support(G, idxs)):
            if debug: print("1", end="", flush=True)
            continue

        if len(in_support(H, idxs)):
            if debug: print("2", end="", flush=True)
            continue

        jdxs = set()
        while len(jdxs) < k:
            jdx = randint(0, n-1)
            if jdx not in idxs:
                jdxs.add(jdx)
        jdxs = list(jdxs)
        jdxs.sort()

        #print(in_support(G, jdxs))
        #print(in_support(H, jdxs))

        if len(in_support(G, jdxs)):
            if debug: print("3", end="", flush=True)
            continue

        if len(in_support(H, jdxs)):
            if debug: print("4", end="", flush=True)
            continue

        break

    print("+\n")

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


def oldmain():

    n = argv.get("n", 10)
    k = argv.get("k", 4)
    dist = argv.get("dist", 2)
    max_tries = argv.get("max_tries", 1000)
    verbose = argv.verbose
    trials = argv.get("trials", 1000)

    count = 0
    fails = 0

    if argv.all_codes:
        gen = all_codes(k, n)
    else:
        gen = (rand2(k, n) for _ in range(trials))
    

    for G in gen:
        assert rank(G) == k
        dG = min_weight(G, dist)
        if dG < dist:
            print("[dG]", end="", flush=True)
            continue

        H = find_kernel(G)
        dH = min_weight(H, dist)
        if dH < dist:
            print("[dH]", end="", flush=True)
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
            if not argv.noassert:
                assert 0
            fails += 1

    print("codes found: %d, fails %d"%(count, fails))


def rand_codes(m, n, trials=1000):
    count = 0
    while count < trials:
        H = rand2(m, n)
        if rank(H) == m:
            yield H
            count += 1


def main():

    n = argv.get("n", 10)
    k = argv.get("k", 4)
    assert 2*k<=n
    m = n-k

    dist = argv.get("dist", 2)
    Hdist = argv.get("Hdist", dist)
    max_tries = argv.get("max_tries", 1000)
    verbose = argv.verbose
    trials = argv.get("trials", 1000)

    count = 0
    fails = 0

    if argv.all_codes:
        gen = all_codes(m, n)
    else:
        gen = rand_codes(m, n, trials)
    
    #assert Hdist == 2

    for H in gen:

        assert rank(H) == m
        dH = min_weight(H)
        if dH < Hdist:
            #print("[dH=%d]"%dH, end="", flush=True)
            continue

        G = find_kernel(H)
        dG = min_weight(G)
        if dG < dist:
            #print("[dG]", end="", flush=True)
            continue

        print("H =")
        print(shortstr(H))
        print("G =")
        print(shortstr(G))

        result = search(G, H, max_tries)
        count += 1
        if result:
            print("\n")
        else:
            print("XXXXXXXXXXXXXXXXXXX FAIL\n")
            if not argv.noassert:
                assert 0
            fails += 1

    print("codes found: %d, fails %d"%(count, fails))


if __name__ == "__main__":

    _seed = argv.get("seed")
    if _seed is not None:
        seed(_seed)
        ra.seed(_seed)

    main()


