#!/usr/bin/env python3

"""
Characterize "robust" property of classical codes.
"""

from random import randint, shuffle

import numpy

from qupy.ldpc.solve import shortstrx, zeros2, array2, dot2, parse, identity2, rank
from qupy.ldpc.solve import find_kernel, cokernel, eq2, get_reductor, intersect
from qupy.ldpc.solve import linear_independent, solve, row_reduce, rand2
from qupy.ldpc.solve import remove_dependent, dependent_rows
from qupy.ldpc.gallagher import make_gallagher, classical_distance



def in_support(H, keep_idxs, check=False): # copied from classical.py
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



def random_code(n, k, kt, distance=1): # copied from puncture.py
    "code length n, dimension k, transpose dimension kt"
    d = 0
    while d<distance:
        H = rand2(n-k, n)
        if rank(H) < n-k:
            continue
        d = classical_distance(H, distance)

    K = H
    dt = 0
    while dt<distance:
        R = rand2(kt, n-k)
        J = dot2(R, H)
        K = numpy.concatenate((H, J))
        if rank(K) < n-k:
            continue
        dt = classical_distance(K.transpose())

    return K




def randremove(items, k):
    # remove k elements from items
    #items = list(items)
    assert len(items)>=k
    found = []
    while len(found) < k:
        idx = randint(0, len(items)-1)
        found.append(items.pop(idx))
    found.sort()
    return found


def is_robust(H, G, trials=1000):
    m, n = H.shape
    k = len(G)
    assert G.shape == (k, n)
    assert m+k == n

    for trial in range(trials):
        remain = list(range(n))
        idxs = randremove(remain, k)
        if len(in_support(G, idxs)):
            continue
        if len(in_support(H, idxs)):
            continue

        jdxs = randremove(remain, k)
        if len(in_support(G, jdxs)):
            continue
        if len(in_support(H, jdxs)):
            continue

        break
    else:
        return None
    
    return idxs, jdxs


def echelon1(A, row, col):
    "Use A[row, col]!=0 to kill all other nonzeros in that col"
    A = A.copy()
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


def has_property(G, trials=1000):
    k, n = G.shape

#    print("has_property")
    Ik = identity2(k)
    for trial in range(trials):
#        print("G =")
#        print(shortstrx(G))
        pivots = []
        remain = list(range(n))
        shuffle(remain)
        G1 = G
        for row in range(k):
          for col in list(remain):
            if G1[row, col] == 0:
                continue
            G1 = echelon1(G1, row, col)
            pivots.append(col)
            remain.remove(col)
            break
        if len(pivots) < k:
            continue
            
        J = G1[:, remain]
#        print("J:")
#        print(shortstrx(J))
        if rank(J)==k:
            break

    else:
        return False

    if 0:
        print("has_property")
        print(G)
        print(G1)
        print(R)
        print()
    return True

fails = """
.1.1.11 11111..
.11.... 1..1.1.
11.11.. 1..1..1
1111...

111.1.. 11111..
...11.. 1.1..1.
.1.1... 1.1...1
1..1.11

"""


def equiv():
    # test that robustness is equiv. to full-rank property

    while 1:
        H = random_code(8,  4, 0, 1) # n, k, kt, d
        #H = random_code(6,  3, 0, 1) # n, k, kt, d
        G = find_kernel(H)
        #if classical_distance(G)==1:
        #    continue

        print()
        print("__"*20)
        print(shortstrx(H, G))

        result = is_robust(H, G)
        lhs = result is not None
        if lhs:
            idxs, jdxs = result
            print(idxs, jdxs)
#        else:
#            print("FAIL")

        rhs = has_property(G, 1000)
        assert lhs==rhs, (lhs, rhs)
        assert not rhs or lhs # rhs ==> lhs
        print(lhs)

        print()



if __name__ == "__main__":

    equiv()


