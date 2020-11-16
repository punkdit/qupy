#!/usr/bin/env python3

"""
Characterize "robust" property of classical codes.
"""

from random import randint, shuffle, seed

import numpy

from qupy.ldpc.solve import shortstrx, zeros2, array2, dot2, parse, identity2, rank
from qupy.ldpc.solve import find_kernel, cokernel, eq2, get_reductor, intersect
from qupy.ldpc.solve import linear_independent, solve, row_reduce, rand2
from qupy.ldpc.solve import remove_dependent, dependent_rows
from qupy.ldpc.gallagher import make_gallagher, classical_distance
from qupy.argv import argv

#from gallagher import make_gallagher


def classical_distance(H, max_dist=0):
    if max_dist==0:
        return max_dist
    n = H.shape[1]
    dist = n
    K = find_kernel(H)
    K = numpy.array(K)
    Kt = K.transpose()
    for u in numpy.ndindex((2,)*K.shape[0]):
        v = dot2(Kt, u)
        if 0 < v.sum() < dist:
            dist = v.sum()
            #if dist<=max_dist:
            #    break
    return dist


def make_gallagher(r, n, l, m, distance=0, verbose=False):
    assert r%l == 0
    assert n%m == 0
    assert r*m == n*l
    if verbose:
        print("make_gallagher", r, n, l, m, distance)
    H = zeros2(r, n)
    H1 = zeros2(r//l, n)
    H11 = identity2(r//l)
    #print(H1)
    #print(H11)
    for i in range(m):
        H1[:, (n//m)*i:(n//m)*(i+1)] = H11
    #print(shortstrx(H1))

    while 1:
        H2 = H1.copy()
        idxs = list(range(n))
        for i in range(l):
            H[(r//l)*i:(r//l)*(i+1), :] = H2
            shuffle(idxs)
            H2 = H2[:, idxs]
        #print(H.shape)
        Hli = linear_independent(H)
        k = Hli.shape[0] - Hli.shape[1]
        assert k <= 24, "ummm, too big? k = %d" % k
        if distance is None:
            break
        if verbose:
            write("/")
        dist = classical_distance(Hli, distance)
        if dist >= distance:
            break
        if verbose:
            write(".")
    if verbose:
        write("\n")
    return Hli




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
    assert len(items)>=k
    found = []
    while len(found) < k:
        idx = randint(0, len(items)-1)
        found.append(items.pop(idx))
    found.sort()
    return found


def get_bipuncture(G, H, trials=1000):
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


def build_echelon(G, H, idxs, jdxs):
    k, n = G.shape
    m, n = H.shape
    assert len(idxs)==len(jdxs)==k
    print("build_echelon")

    G0 = G # save 
    pivots = []
    for col in idxs:
        #print("col:", col)
        #print(shortstrx(G))
        for row in range(k):
            if G[row, col]:
                pivots.append((row, col))
                G = echelon1(G, row, col)
                break
        else:
            assert 0
    for row, col in pivots:
        assert G[row, col]

    H0 = H # save 
    pivots = []
    for col in jdxs:
        #print("col:", col)
        #print(shortstrx(H))
        for row in range(k):
            if H[row, col]:
                pivots.append((row, col))
                H = echelon1(H, row, col)
                break
        else:
            assert 0
    for row, col in pivots:
        assert H[row, col]



def has_property(G, trials=1000):
    k, n = G.shape

    for trial in range(trials):
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
        if rank(J)==k:
            break
    else:
        return False
    return True


def main():
    # test that robustness is equiv. to full-rank property

    #n, k, kt, d = 8, 4, 0, 1

    cw = argv.get("cw", 3) # column weight
    rw = argv.get("rw", 4) # row weight
    n = argv.get("n", 8) # cols
    m = argv.get("m", n*cw//rw) # rows
    d = argv.get("d", 1) # distance
    rank = argv.get("rank", 0)

    print("m =", m)

    trials = argv.get("trials", 1000)

    while 1:
        #H = random_code(n, k, kt, d)
        H = make_gallagher(m, n, cw, rw, d)

        if len(H) < rank:
            continue

        G = find_kernel(H)

        print()
        print("__"*20)
        print("G:%sH:"%(' '*(n-1),))
        print(shortstrx(G, H))
        print(G.shape, H.shape)

        result = get_bipuncture(G, H, trials)
        print("get_bipuncture:", result)
        robust = result is not None
        if robust:
            idxs, jdxs = result
            build_echelon(G, H, idxs, jdxs)

        rhs = has_property(G, trials)
        assert robust==rhs, (robust, rhs)
        #assert not rhs or robust # rhs ==> robust
        #print(robust)

        print()

        if argv.robust:
            assert robust



if __name__ == "__main__":
    _seed = argv.get("seed")
    if _seed is not None:
        seed(_seed)
        numpy.random.seed(_seed)

    main()


