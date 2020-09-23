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
    # find span of H _contained within idxs support
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



def search(G, H, max_tries=None, size=None, debug=False):

    m, n = H.shape
    k, n1 = G.shape
    assert n==n1
    assert m+k==n

    check = shortstr(H)+shortstr(G)

    size = size or k

    count = 0
    while 1:
        assert check == shortstr(H)+shortstr(G)
        count += 1
        if max_tries is not None and count>max_tries:
            #print("/\n")
            return False

        idxs = set()
        while len(idxs) < size:
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
        while len(jdxs) < size:
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

    #print("+\n")

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


#def oldmain():
#
#    n = argv.get("n", 10)
#    k = argv.get("k", 4)
#    dist = argv.get("dist", 2)
#    max_tries = argv.get("max_tries", 1000)
#    verbose = argv.verbose
#    trials = argv.get("trials", 1000)
#
#    count = 0
#    fails = 0
#
#    if argv.all_codes:
#        gen = all_codes(k, n)
#    else:
#        gen = (rand2(k, n) for _ in range(trials))
#    
#
#    for G in gen:
#        assert rank(G) == k
#        dG = min_weight(G, dist)
#        if dG < dist:
#            print("[dG]", end="", flush=True)
#            continue
#
#        H = find_kernel(G)
#        dH = min_weight(H, dist)
#        if dH < dist:
#            print("[dH]", end="", flush=True)
#            continue
#
#        print("G =")
#        print(shortstr(G))
#        print("H =")
#        print(shortstr(H))
#        result = search(G, H, max_tries)
#        count += 1
#        if result:
#            print("\n")
#        else:
#            print("XXXXXXXXXXXXXXXXXXX FAIL\n")
#            if not argv.noassert:
#                assert 0
#            fails += 1
#
#    print("codes found: %d, fails %d"%(count, fails))
#

def rand_codes(m, n, trials=10000):
    count = 0
    while count < trials:
        H = rand2(m, n)
        if rank(H) == m:
            yield H
            count += 1


def weight_dist(H):
    m, n = H.shape
    counts = {i:0 for i in range(n+1)}
    #print(m, n, counts)
    for u in numpy.ndindex((2,)*m):
        v = numpy.dot(u, H) % 2
        d = v.sum()
        #print(u, v, d)
        counts[d] += 1
    return [counts[i] for i in range(n)]

A = array2([[1,1,0],[0,1,1]])
assert weight_dist(A) == [1, 0, 3]


def get_codes(m, n):
    while 1:
        H = rand2(m, n)
        #H[0:2, :] = 0
        H[:, 0] = 0
        H[0, 0] = 1
        #H[0:2, 0:3] = A
        #print(shortstr(H))
        #print()
        if rank(H) < m:
            continue
        yield H



def main():

    n = argv.get("n", 8)
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
    elif argv.get_codes:
        gen = get_codes(m, n)
    elif argv.gallagher:
        cw = argv.get("cw", 3) # column weight
        rw = argv.get("rw", 4) # row weight
        n = argv.get("n", 12) # cols
        m = argv.get("m", n*cw//rw) # rows
        def gen(trials=1000):
            for _ in range(trials):
                H = make_gallagher(m, n, cw, rw, dist)
                #if rank(H) < m:
                #    continue
                #print(shortstr(H))
                yield H
        gen = gen(trials)

    elif argv.wedge:
        H = zeros2(m, n)
        H[0, :k+1] = 1
        H[:, k-1] = 1
        for i in range(m):
            H[i, i+k] = 1
        gen = [H]
        #print(shortstr(H))
        #print()

    elif argv.cookup:
        # fail
        gen = [parse("""
1111........
..1.1.......
..1..1......
..1...1.....
..1....1....
..1.....1...
..1......1..
..1.......1.
..1........1
        """)]

    else:
        gen = rand_codes(m, n, trials)
    
    #assert Hdist == 2

    for H in gen:

        #assert rank(H) == m
        dH = min_weight(H)
        if dH < Hdist:
            #print("[dH=%d]"%dH, end="", flush=True)
            continue

        G = find_kernel(H)
        #print(shortstr(G))
        dG = min_weight(G)
        if dG < dist:
            #print("[dG=%d]"%dG, end="", flush=True)
            continue

        result = search(G, H, max_tries)
        count += 1
        if result:
            print(".", end="", flush=True)
            continue
        print()

        for size in range(2, k):
            result = search(G, H, max_tries, size=size)
            print("result(size=%d)"%size, result)

        process(G, H)

        #print("XXXXXXXXXXXXXXXXXXX FAIL\n")
        if not argv.noassert:
            assert 0, "FAIL"
        fails += 1

    print()
    print("codes found: %d, fails %d"%(count, fails))




def swap_row(A, j, k):
    row = A[j, :].copy()
    A[j, :] = A[k, :]
    A[k, :] = row

def swap_col(A, j, k):
    col = A[:, j].copy()
    A[:, j] = A[:, k]
    A[:, k] = col


def echelon(A, row, col):
    #A = A.copy()
    m, n = A.shape
    assert A[row, col] != 0
    for i in range(m):
        if i==row:
            continue
        if A[i, col]:
            A[i, :] += A[row, :]
            A %= 2
            assert A[i, col] == 0
    return A


def process(G, H):

    #print("H =")
    #print(shortstr(H))
    #print("G =")
    #print(shortstr(G))

    m, n = G.shape
    row = 0
    while row < m:

        col = 0
        while G[row, col] == 0:
            col += 1
        assert col >= row
        swap_col(G, row, col)
        swap_col(H, row, col)
        echelon(G, row, row)

        row += 1

    col = row
    row = 0
    m, n = H.shape
    while row < m:
        j = col
        while H[row, j] == 0:
            j += 1
        swap_col(G, col, j)
        swap_col(H, col, j)
        echelon(H, row, col)

        row += 1
        col += 1


    print("G =")
    print(shortstr(G))
    print("H =")
    print(shortstr(H))

    print("H:", weight_dist(H))
    print("G:", weight_dist(G))
    #print("Ht:", weight_dist(H.transpose()))
    #print("Gt:", weight_dist(G.transpose()))
    print()


if __name__ == "__main__":

    _seed = argv.get("seed")
    if _seed is not None:
        seed(_seed)
        ra.seed(_seed)

    main()


