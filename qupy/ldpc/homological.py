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
    



def rand_rowspan(A): # rowspan !
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
    #KerA = rand_rowspan(KerA) # ??
    KerA = row_reduce(KerA)
    ka = len(KerA)

    assert KerA.shape[1] == na
    K = KerA.transpose()
    #K = rand_rowspan(K)
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
    # once we add stabilizers it will be reduced:
    assert rank(Lz) == len(Lz)

    # ---------------------------------------------------
    # Build Lx 

    counit = lambda n : unit2(n).transpose()

    CokerB = find_cokernel(B) # matrix of row vectors
    #CokerB = rand_rowspan(CokerB)
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


def test(A, B, ma, na, mb, nb, Ina, Ima, Inb, Imb, ka, kb, kat, kbt, k,
    KerA, KerB, CokerA, CokerB,
    Lzi, Lxi, Hzi, Hxi,
    **kw):

    #print("ka=%s, kat=%s, kb=%s, kbt=%s"%(ka, kat, kb, kbt))
    assert k == ka*kbt + kat*kb == len(Lzi) == len(Lxi)

    KerA = KerA.transpose() # use convention in paper
    KerB = KerB.transpose() # use convention in paper
    CokerA = CokerA.transpose() # use convention in paper
    CokerB = CokerB.transpose() # use convention in paper

    blocks = [
        [kron(KerA, Imb), zeros2(na*mb, ma*kb), kron(Ina, B)],
        [zeros2(ma*nb, ka*mb), kron(Ima, KerB), kron(A,Inb)],
    ]
    print("blocks:", [[X.shape for X in row] for row in blocks])

    #print(shortstrx(*blocks[0]))
    #print()
    #print(shortstrx(*blocks[1]))

    Hzt = cat((blocks[0][2], blocks[1][2]), axis=0)
    K = find_kernel(Hzt)
    assert len(K) == ka*kb # see proof of Lemma 3

    Lzv = cat((blocks[0][0], blocks[1][0])).transpose()
    Lzh = cat((blocks[0][1], blocks[1][1])).transpose()
    assert dot2(Hxi, Lzv.transpose()).sum() == 0

#    Hz = Hzt.transpose()
#    Hzi = linear_independent(Hz)
#    Lzhi = independent_logops(Lzh, Hzi, verbose=True)
#    print("Lzhi:", Lzhi.shape)

    # --------------------------------------------------------
    # basis for all logops, including stabilizers
    lz = find_kernel(Hxi) # returns transpose of kernel
    #lz = rand_rowspan(lz)
    #print("lz:", lz.shape)
    assert len(lz) == k+len(Hzi)

    # vertical qubits
    Iv = cat((identity2(na*mb), zeros2(ma*nb, na*mb)), axis=0).transpose()
    # horizontal qubits
    Ih = cat((zeros2(na*mb, ma*nb), identity2(ma*nb)), axis=0).transpose()
    assert len(intersect(Iv, Ih))==0 # sanity check

    # now restrict these logops to vertical qubits
    #print("Iv:", Iv.shape)
    lzv = intersect(Iv, lz)
    #print("lzv:", lzv.shape)

    J = intersect(lzv, Lzv)
    assert len(J) == len(lzv)

    # --------------------------------------------------------
    # now we manually build _lz supported on vertical qubits
    x = rand2(ka*mb, ka*nb)
    y = kron(KerA, Inb)
    assert eq2(dot2(blocks[0][2], y), kron(KerA, B))
    v = (dot2(blocks[0][0], x) + dot2(blocks[0][2], y)) % 2
    h = zeros2(ma*nb, v.shape[1])
    _lzt = cat((v, h))
    assert dot2(Hxi, _lzt).sum() == 0
    #print(shortstr(_lzt))
    _lz = _lzt.transpose()
    _lz = linear_independent(_lz)

    #print("*"*(na*mb))
    #print(shortstr(_lz))
    assert len(intersect(_lz, Ih)) == 0
    assert len(intersect(_lz, Iv)) == len(_lz)

    J = intersect(_lz, lz)
    assert len(J) == len(_lz)

    J = intersect(_lz, Lzv)
    #print(J.shape, _lz.shape, Lzv.shape)
    assert len(J) == len(_lz)

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
    

def get_puncture(M, k):
    "k-puncture the rowspace of M"
    m, n = M.shape

    assert 0<=k<=n
    mask = [1]*k + [0]*(n-k)

    while 1:
        shuffle(mask)
        a = array2(mask)
        a.shape = (n, 1)
        u = solve(M.transpose(), a)
        if u is None:
            break
    idxs = [i for i in range(n) if mask[i]]
    return idxs


def test_puncture(A, B, ma, na, mb, nb, Ina, Ima, Inb, Imb, ka, kb, kat, kbt, k,
    KerA, KerB, CokerA, CokerB,
    Lzi, Lxi, Hzi, Hxi,
    **kw):

    I = identity2

    assert ka - na + ma -kat == 0 
    assert kb - nb + mb -kbt == 0 

    #print("ka=%s, kat=%s, kb=%s, kbt=%s"%(ka, kat, kb, kbt))
    assert k == ka*kbt + kat*kb == len(Lzi) == len(Lxi)

    kernel = lambda X : find_kernel(X).transpose() # use convention in paper
    KerA = KerA.transpose() # use convention in paper
    KerB = KerB.transpose() # use convention in paper
    #CokerA = CokerA.transpose() # use convention in paper
    #CokerB = CokerB.transpose() # use convention in paper

    assert CokerA.shape == (kat, ma)
    assert CokerB.shape == (kbt, mb)

    blocks = [
        [kron(KerA, Imb), zeros2(na*mb, ma*kb), kron(Ina, B)],
        [zeros2(ma*nb, ka*mb), kron(Ima, KerB), kron(A,Inb)],
    ]
    print("blocks:", [[X.shape for X in row] for row in blocks])

    #print(shortstrx(*blocks[0]))
    #print()
    #print(shortstrx(*blocks[1]))

    Mv = cat((blocks[0][0], blocks[0][2]), axis=1)
    Mh = cat((blocks[1][0], blocks[1][2]), axis=1)
    M = cat((Mv, Mh), axis=0)
    KM = kernel(M)

    Mv = cat(blocks[0], axis=1)
    Mh = cat(blocks[1], axis=1)
    M = cat((Mv, Mh), axis=0)

    x = kron(I(ka), B)
    dot2(blocks[0][0], x)

    y = zeros2(blocks[0][1].shape[1], x.shape[1])
    dot2(blocks[0][1], y)

    z = kron(KerA, I(nb))
    dot2(blocks[0][2], z)

    #print(shortstr(x)+'\n')
    #print(shortstr(y)+'\n')
    #print(shortstr(z)+'\n')

    xz = cat((x, z), axis=0)
    xyz = cat((x, y, z), axis=0)
    assert dot2(M, xyz).sum() == 0
    #print(shortstr(xyz))
    print("xyz:", xyz.shape)
    assert len(find_kernel(xyz))==0

    assert rowspan_eq(KM.transpose(), xz.transpose())

    print("kernel(M):", kernel(M).shape)

    Hzt = cat((blocks[0][2], blocks[1][2]), axis=0)
    print("kernel(Hzt):", kernel(Hzt).shape)
    Hx = cat((kron(A, I(mb)), kron(I(ma), B)), axis=1)

    #print("CokerB")
    #print(shortstr(CokerB))

    #R = CokerB
    #R = rand2(CokerB.shape[0], CokerB.shape[1])
    #R = rand2(mb, 1)
    #R = CokerB[:, 0:1]

    if 1:
        idxs = get_puncture(B, kbt)
        R = zeros2(mb, 1)
        R[idxs] = 1
    else:
        R = B[:, :1]

    #R = rand2(mb, 100)
    #R = I(mb)
    lzt = cat((kron(KerA, R), zeros2(ma*nb, KerA.shape[1]*R.shape[1])), axis=0)

    assert dot2(Hx, lzt).sum()==0

    lz = lzt.transpose()
    Hz = Hzt.transpose()
    print(rank(lz), rank(Hz), rank(intersect(lz, Hz)))

    print(rowspan_le(lzt.transpose(), Hzt.transpose())) # FAIL
    #assert rowspan_le(lzt.transpose(), Hzt.transpose()) # FAIL

    print("OK")


def rowspan_le(A, B):
    u = solve(B.transpose(), A.transpose())
    if u is None:
        return False
    return True


def rowspan_eq(A, B):
    return rowspan_le(A, B) and rowspan_le(B, A)


def unit_test():
    A = array2([[1,1,1]])
    B = array2([[1,1,0],[0,0,1]])
    C = array2([[0,1,0]])
    assert rowspan_le(A, B)
    assert not rowspan_le(B, A)
    assert not rowspan_le(A, C)
    assert not rowspan_le(B, C)
unit_test()


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
    "return parity check for code of length n, dimension k, transpose dimension kt"
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

    m = len(K)
    assert k - n + m - kt == 0 

    return K


def main():

    n = argv.get("n", 8)
    k = argv.get("k", 4)
    kt = argv.get("kt", 4)
    d = argv.get("d", 1) # distance
    na = argv.get("na", n)
    nb = argv.get("nb", n)
    ka = argv.get("ka", k)
    kat = argv.get("kat", kt)
    kb = argv.get("kb", k)
    kbt = argv.get("kbt", kt)
    A = random_code(na, ka, kat, d)
    B = random_code(nb, kb, kbt, d)
    #assert A.shape == (na-ka, na), (A.shape,)
    #assert B.shape == (nb-kb, nb), (B.shape,)

    print("A, B:")
    print(shortstrx(A, B))

    if 1:
        # A tensor B
        kw = hypergraph_product(A, B)
        test_puncture(**kw)

    else:
        # --------------------------------------
        #B, Bt = Bt, B
    
        KerA = find_kernel(A).transpose()
        ma = na-ka
        mb = nb-kb
    
        print("KerA:")
        print(shortstrx(KerA))
        print()
    #    print(shortstrx(kron(KerA, identity2(mb))))
    
        
        print(shortstrx(
            kron(identity2(mb), KerA), 
            kron(B, identity2(na))))
    
    
    

if __name__ == "__main__":

    _seed = argv.get("seed")
    if _seed is not None:
        seed(_seed)
        ra.seed(_seed)

    while 1:
        main()
        if not argv.forever:
            break


