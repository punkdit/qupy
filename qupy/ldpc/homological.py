#!/usr/bin/env python3

"""
copied from : puncture.py 

renaming C/D as A/B, etc.

"""

import sys
from random import randint, seed, random, shuffle

import numpy
import numpy.random as ra
from numpy.linalg import lstsq
cat = numpy.concatenate

from qupy.argv import argv
from qupy.tool import write, choose
from qupy.ldpc.css import CSSCode, randcss
from qupy.ldpc.solve import shortstrx, zeros2, array2, dot2, parse, identity2, rank
from qupy.ldpc.solve import find_kernel, cokernel, eq2, get_reductor, intersect
from qupy.ldpc.solve import linear_independent, solve, row_reduce, rand2
from qupy.ldpc.solve import remove_dependent, dependent_rows
from qupy.ldpc.gallagher import make_gallagher, classical_distance
from qupy.ldpc.main import get_decoder

shortstr = shortstrx


def find_cokernel(H):
    P, _ = cokernel(H)
    return P


def unit2(n, i=0):
    e = zeros2(n, 1)
    e[i, 0] = 1
    return e


def kron(A, B):
    if 0 in A.shape or 0 in B.shape:
        C = zeros2(A.shape[0]*B.shape[0], A.shape[1]*B.shape[1])
    else:
        #print("kron", A.shape, B.shape)
        C = numpy.kron(A, B)
        #print("\t", C.shape)
    return C


def independent_logops(L, H, verbose=False):
    m = len(H)
    LH = numpy.concatenate((L, H), axis=0)
    keep, remove = dependent_rows(LH)
    if verbose:
        print(keep, remove)
    LH = LH[keep]
    
    assert rank(LH) == len(LH)
    assert len(LH) >= m
    assert eq2(LH[-len(H):], H), "H not linearly independent ?"
    L = LH[:-m]

    return L

    
def get_k(L, H):
    return len(independent_logops(L, H))
    

def min_span(K):
    "find minimum weight span"
    Kt = K.transpose()
    dist = {}
    for u in numpy.ndindex((2,)*K.shape[0]):
        v = dot2(Kt, u)
        if v.sum()==0:
            continue
        weight = v.sum()
        #dist[weight] = dist.get(weight, 0) + 1
        dist.setdefault(weight, []).append(v)
    keys = list(dist.keys())
    keys.sort(reverse=True)
    rows = []
    for k in keys:
        #print("%s:%s" % (k, len(dist[k])), end=" ")
        rows.extend(dist[k])
    #print()
    A = array2(rows)
    #print(A.shape)
    A = remove_dependent(A)
    #print(shortstr(A))
    return A


def rand_span(A): # rowspan !
    while 1:
        m, n = A.shape
        v = rand2(m, m)
        A1 = dot2(v, A)
        assert A1.shape == A.shape
        if rank(A) == rank(A1):
            break
    assert rank(intersect(A, A1)) == rank(A)
    return A1


def hypergraph_product(A, B, check=False):
    print("hypergraph_product: A=%s, B=%s"%(A.shape, B.shape))

    ma, na = A.shape
    mb, nb = B.shape

    Ima = identity2(ma)
    Imb = identity2(mb)
    Ina = identity2(na)
    Inb = identity2(nb)

    Hz0 = kron(Ina, B.transpose()), kron(A.transpose(), Inb)
    Hz = numpy.concatenate(Hz0, axis=1) # horizontal concatenate

    Hx0 = kron(A, Imb), kron(Ima, B)
    #print("Hx0:", Hx0[0].shape, Hx0[1].shape)
    Hx = numpy.concatenate(Hx0, axis=1) # horizontal concatenate

    assert dot2(Hx, Hz.transpose()).sum() == 0

    n = Hz.shape[1]
    assert Hx.shape[1] == n

    # ---------------------------------------------------
    # Build Lz 

    KerA = find_kernel(A)
    #KerA = min_span(KerA) # does not seem to matter... ??
    KerA = rand_span(KerA) # ??
    KerA = row_reduce(KerA)
    ka = len(KerA)

    assert KerA.shape[1] == na
    K = KerA.transpose()
    #K = min_span(K)
    #K = rand_span(K)
    #E = identity2(mb)

    #print("ma,na,mb,nb=", ma, na, mb, nb)

    Lzt0 = kron(K, Imb), zeros2(ma*nb, K.shape[1]*mb)
    Lzt0 = numpy.concatenate(Lzt0, axis=0)
    assert dot2(Hx, Lzt0).sum() == 0

    KerB = find_kernel(B)
    KerB = row_reduce(KerB)
    kb = len(KerB)
    K = KerB.transpose()
    assert K.shape[0] == nb

    Lzt1 = zeros2(na*mb, K.shape[1]*ma), kron(Ima, K)
    Lzt1 = numpy.concatenate(Lzt1, axis=0)
    assert dot2(Hx, Lzt1).sum() == 0

    Lzt = numpy.concatenate((Lzt0, Lzt1), axis=1) # horizontal concatenate
    Lz = Lzt.transpose()

    assert dot2(Hx, Lzt).sum() == 0

    # These are linearly independent among themselves, but 
    # once we add stabilixers it will be reduced:
    assert rank(Lz) == len(Lz)

    # ---------------------------------------------------
    # Build Lx 

    counit = lambda n : unit2(n).transpose()

    CokerB = find_cokernel(B) # matrix of row vectors
    #CokerB = min_span(CokerB)
    CokerB = rand_span(CokerB)
    assert rank(CokerB)==len(CokerB)
    kbt = len(CokerB)

    Lx0 = kron(Ina, CokerB), zeros2(CokerB.shape[0]*na, ma*nb)
    Lx0 = numpy.concatenate(Lx0, axis=1) # horizontal concatenate

    assert dot2(Lx0, Hz.transpose()).sum() == 0

    CokerA = find_cokernel(A)
    assert rank(CokerA)==len(CokerA)
    kat = len(CokerA)

    Lx1 = zeros2(CokerA.shape[0]*nb, na*mb), kron(CokerA, Inb)
    Lx1 = numpy.concatenate(Lx1, axis=1) # horizontal concatenate

    Lx = numpy.concatenate((Lx0, Lx1), axis=0)

    assert dot2(Lx, Hz.transpose()).sum() == 0

    #print(ka, kat, kb, kbt)

    # ---------------------------------------------------
    # 

    overlap = 0
    for lz in Lz:
      for lx in Lx:
        w = (lz*lx).sum()
        overlap = max(overlap, w)
    assert overlap <= 1, overlap
    #print("max overlap:", overlap)

    if 0:
        # here we assume that Hz/Hx are full rank
        Hzi = Hz
        Hxi = Hx
        assert rank(Hz) == len(Hz)
        assert rank(Hx) == len(Hx)
        mz = len(Hz)
        mx = len(Hx)
    else:
        Hzi = remove_dependent(Hz)
        Hxi = remove_dependent(Hx)
        mz = rank(Hz)
        mx = rank(Hx)
        assert len(Hzi) == mz
        assert len(Hxi) == mx


    # ---------------------------------------------------
    # 

    Lz0, Lz1 = Lzt0.transpose(), Lzt1.transpose()

    Lzi = independent_logops(Lz, Hzi)
    Lxi = independent_logops(Lx, Hxi)

    print("Lzi:", len(Lzi))
    print("Lxi:", len(Lxi))

    k = len(Lzi)
    assert len(Lxi) == k
    assert mz + mx + k == n

    LziHz = numpy.concatenate((Lzi, Hzi))
    assert rank(LziHz) == k+mz
    LxiHx = numpy.concatenate((Lxi, Hxi))
    assert rank(LxiHx) == k+mx

    return locals()


def test(A, B, ma, na, mb, nb, Ina, Ima, Inb, Imb, ka, kb, kat, kbt, 
    KerA, KerB, CokerA, CokerB,
    **kw):

    print("ka=%s, kat=%s, kb=%s, kbt=%s"%(ka, kat, kb, kbt))

    blocks = [
        [kron(KerA.transpose(), Imb), zeros2(na*mb, ma*kb), kron(Ina, B)],
        [zeros2(ma*nb, ka*mb), kron(Ima, KerB.transpose()), kron(A,Inb)],
    ]
    print("blocks:", [[X.shape for X in row] for row in blocks])

    Hzt = cat((blocks[0][2], blocks[1][2]), axis=0)
    K = find_kernel(Hzt)
    assert len(K) == ka*kb # see proof of Lemma 3

    Lzv = cat((blocks[0][0], blocks[1][0])).transpose()
    Lzh = cat((blocks[0][1], blocks[1][1])).transpose()

    Hz = Hzt.transpose()
    Hzi = linear_independent(Hz)
    Lzhi = independent_logops(Lzh, Hzi, verbose=True)
    print("Lzhi:", Lzhi.shape)

    if 0:
        V = cat(blocks[0][:2], axis=1)
        H = cat(blocks[1][:2], axis=1)
        X = cat((V, H), axis=0)
        K = find_kernel(X)
        print(K.shape)
    
        V = cat(blocks[0], axis=1)
        H = cat(blocks[1], axis=1)
        X = cat((V, H), axis=0)
        K = find_kernel(X)
        print(K.shape)
    
        #print("-"*(ka*mb+ma*kb))
        I = cat((identity2(ka*mb+ma*kb), zeros2(ka*mb+ma*kb, na*nb)), axis=1)
        J = intersect(K, I)
        print("J:", J.shape)
    

def eq_span(A, B):
    u = solve(A.transpose(), B.transpose())
    if u is None:
        return False
    u = solve(B.transpose(), A.transpose())
    return u is not None


def shuff2(A):
    idxs = list(range(len(A)))
    shuffle(idxs)
    A = A[idxs]
    return A
    

def shuff22(A):
    m, n = A.shape
    idxs = list(range(m))
    shuffle(idxs)
    A = A[idxs]
    idxs = list(range(n))
    shuffle(idxs)
    A = A[:, idxs]
    return A

    
def get_pivots(H):
    m, n = H.shape
    i = j = 0
    items = []
    while i < m:
        while H[i, j] == 0:
            j += 1
            assert j<n
        items.append((i, j))
        i += 1
    return items


def random_code(n, k, kt, distance=1):
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


def main():

    if argv.ldpc:
        # LDPC
        l = argv.get("l", 3) # column weight
        m = argv.get("m", 4) # row weight
        n = argv.get("n", 8) # cols
        r = argv.get("r", n*l//m) # rows
        d = argv.get("d", 1) # distance
        print("make_gallagher%s"%((r, n, l, m, d),))
        A = make_gallagher(r, n, l, m, d)
        print(shortstr(A))
        print()
        print(shortstr(A))
        print("rank(A) = ", rank(A), "kernel(A) = ", len(find_kernel(A)))
        if argv.same:
            B = A
        else:
            B = make_gallagher(r, n, l, m, d)
            assert rank(A) == len(A)
            assert rank(B) == len(B)
            print("rank(B)", rank(B), "kernel(B)", len(find_kernel(B)))

    elif argv.rand:
        # restrict to the vertical sector
        na = argv.get("na", 8)
        nb = argv.get("nb", 8)
        ka = argv.get("ka", 4)
        kat = argv.get("kat", 4)
        kb = argv.get("kb", 4)
        kbt = argv.get("kbt", 4)
        A = random_code(na, ka, kat, 3)
        B = random_code(nb, kbt, kb, 3)

    elif argv.vrand:
        A = random_code(16, 8, 1, 3)
        B = random_code(8,  4, 1, 3)

    elif argv.hvrand:
        #A = random_code(16, 8, 8, 3)
        A = random_code(8,  4, 4, 3)
        B = random_code(8,  4, 4, 3)

    elif argv.pair:
        #A = make_gallagher(9, 12, 3, 4, 4) # big
        A = make_gallagher(15, 20, 3, 4, 4) # big
        B = make_gallagher(6, 8, 3, 4, 1) # small

    elif argv.torus:
        # Torus
        A = parse("""
        11..
        .11.
        ..11
        1..1
        """)
        B = A

    elif argv.hamming:
        A = parse("""
        ...1111
        .11..11
        1.1.1.1
        111....
        """)
        B = A

    elif argv.surf or argv.surface:
        # Surface
        A = parse("""
        11....
        .11...
        ..11..
        ...11.
        ....11
        """)
        B = parse("""
        11..
        .11.
        ..11
        """)

    elif argv.small:
        A = parse("""1111""")
        B = parse("""1111""")

    else:
        print("please specify a code")
        return

    print("A: shape=%s, rank=%d, dist=%d"%(A.shape, rank(A), classical_distance(A)))
    print("A.t: dist=%d"%(classical_distance(A.transpose()),))
    print(shortstr(A))
    print("B: shape=%s, rank=%d, dist=%d"%(B.shape, rank(B), classical_distance(B)))
    print("B.t: dist=%d"%(classical_distance(B.transpose()),))
    print(shortstr(B))
    At = A.transpose()
    Bt = B.transpose()

    # A tensor Bt
    kw = hypergraph_product(A, Bt)

    test(**kw)


if __name__ == "__main__":

    _seed = argv.get("seed")
    if _seed is not None:
        seed(_seed)
        ra.seed(_seed)

    while 1:
        main()
        if not argv.forever:
            break


