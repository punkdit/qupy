#!/usr/bin/env python3

from random import randint, seed

import numpy
import numpy.random as ra
from numpy.linalg import lstsq

from qupy.argv import argv
from qupy.tool import write, choose
from qupy.ldpc.css import CSSCode, randcss
from qupy.ldpc.solve import shortstr, zeros2, array2, dot2, parse, identity2, rank
from qupy.ldpc.solve import find_kernel, cokernel, eq2, get_reductor
from qupy.ldpc.solve import linear_independent, solve, row_reduce
from qupy.ldpc.gallagher import make_gallagher, classical_distance


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


def dependent_rows(H):
    "remove dependent rows of H, first to last"
    idxs = set(range(len(H)))
    #print(H)
    K = find_kernel(H.transpose())
    #print("K:")
    #print(K)
    K = row_reduce(K, truncate=True)
    #print("K:")
    #print(K)
    assert dot2(K, H).sum() == 0
    deps = []
    for row in K:
        #print(row)
        idx = numpy.where(row!=0)[0][0]
        deps.append(idx)
        idxs.remove(idx)
    assert len(set(deps)) == len(K)
    idxs = list(idxs)
    idxs.sort()
    deps.sort()
    return idxs, deps


def remove_dependent(H):
    "remove dependent rows of H, first to last"
    if len(H) <2:
        return H
    idxs, deps = dependent_rows(H)
    return H[idxs]


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


def is_correctable(n, idxs, Lx, Lz):
    "is an X error supported on indexes idxs, or Z error on idxs, correctable?"
    A = zeros2(len(idxs), n)
    for i, idx in enumerate(idxs):
        A[i, idx] = 1
    #print("correctable", len(idxs))
    
    for Lops in [Lx, Lz]:
        B = numpy.concatenate((Lops, A), axis=0)
        #print(B.shape)
        B1 = remove_dependent(B)
        #print(B1.shape)
    
        if len(B1) < len(B):
            return False

    return True



def mk_logops(L, H):
    m = len(H)
    #print("L:", len(L))

    L0 = L # save this
    LH = numpy.concatenate((L, H), axis=0)

    #LH = remove_dependent(LH)
    keep, remove = dependent_rows(LH)
    #print("dependent_rows:", len(keep), len(remove))

    LH = LH[keep]

    assert rank(LH) == len(LH)
    assert len(LH) >= m
    assert eq2(LH[-len(H):], H)
    L = LH[:-m]
    #print("L:", len(L))

    # find disjoint set of L ops
    keep = set([idx for idx in keep if idx<len(L0)])
    assert len(keep) == len(L)

    idxs = list(range(len(L0)))
    idxs.sort(key = lambda idx: -int(idx in keep))
    assert idxs[0] in keep
    L1 = L0[idxs]
    assert L1.shape == L0.shape

    LH = numpy.concatenate((L1, H), axis=0)
    LH = remove_dependent(LH)
    L1 = LH[:-m]
    assert len(L) == len(L1), (L.shape, L1.shape)

    #print(shortstr(L))
    #print("--")
    #print(shortstr(L1))

    #print("--")
    A = numpy.dot(L, L1.transpose())
    #assert A.sum() == 0, "found overlap"
    if A.sum():
        print("WARNING: A.sum() =", A.sum())
        print("failed to find disjoint logops")

    return L, L1


def hypergraph_product(C, D, check=False):
    print("hypergraph_product:", C.shape, D.shape)
    print("distance:", classical_distance(C))

    c0, c1 = C.shape
    d0, d1 = D.shape
    E1 = identity2(c0)
    E2 = identity2(d0)
    M1 = identity2(c1)
    M2 = identity2(d1)

    Hx0 = kron(M1, D.transpose()), kron(C.transpose(), M2)
    Hx = numpy.concatenate(Hx0, axis=1) # horizontal concatenate

    Hz0 = kron(C, E2), kron(E1, D)
    #print("Hz0:", Hz0[0].shape, Hz0[1].shape)
    Hz = numpy.concatenate(Hz0, axis=1) # horizontal concatenate

    assert dot2(Hz, Hx.transpose()).sum() == 0

    n = Hx.shape[1]
    assert Hz.shape[1] == n

    # ---------------------------------------------------
    # Build Lx 

    Kt = find_kernel(C)
    #Kt = min_span(Kt) # does not seem to matter... ??
    assert Kt.shape[1] == c1
    K = Kt.transpose()
    #E = unit2(d0)
    E = identity2(d0)

    Lxt0 = kron(K, E), zeros2(c0*d1, K.shape[1]*d0)
    Lxt0 = numpy.concatenate(Lxt0, axis=0)
    assert dot2(Hz, Lxt0).sum() == 0

    K = find_kernel(D).transpose()
    assert K.shape[0] == d1
    #E = unit2(c0)
    E = identity2(c0)

    Lxt1 = zeros2(c1*d0, K.shape[1]*c0), kron(E, K)
    Lxt1 = numpy.concatenate(Lxt1, axis=0)
    assert dot2(Hz, Lxt1).sum() == 0

    #print("Lxt0", Lxt0.shape)
    #print("Lxt1", Lxt1.shape)
    Lxt = numpy.concatenate((Lxt0, Lxt1), axis=1) # horizontal concatenate
    Lx = Lxt.transpose()

    assert dot2(Hz, Lxt).sum() == 0

    #print("Lx:", Lx.shape)
    #print(shortstr(Lx))

    if 0:
        P = get_reductor(Hx)
        Lx = dot2(Lx, P.transpose())
        Lx = linear_independent(Lx)

    # ---------------------------------------------------
    # Build Lz 

    counit = lambda n : unit2(n).transpose()

    K = find_cokernel(D) # matrix of row vectors
    #E = counit(c1)
    E = identity2(c1)
    Lz0 = kron(E, K), zeros2(K.shape[0]*c1, c0*d1)
    Lz0 = numpy.concatenate(Lz0, axis=1) # horizontal concatenate

    assert dot2(Lz0, Hx.transpose()).sum() == 0

    K = find_cokernel(C)
    #E = counit(d1)
    E = identity2(d1)
    Lz1 = zeros2(K.shape[0]*d1, c1*d0), kron(K, E)
    Lz1 = numpy.concatenate(Lz1, axis=1) # horizontal concatenate

    Lz = numpy.concatenate((Lz0, Lz1), axis=0)

    assert dot2(Lz, Hx.transpose()).sum() == 0


    if 0:
        print("Hx:", Hx.shape)
        print(shortstr(Hx))
        print("Hz:", Hz.shape)
        print(shortstr(Hz))
        print("Lx:", Lx.shape)
        print(shortstr(Lx))
        print("Lz:", Lz.shape)
        print(shortstr(Lz))
    
        print("dot2(Lz, Lxt):")
        A = dot2(Lz, Lxt)
        print(shortstr(A))

    overlap = 0
    for lx in Lx:
      for lz in Lz:
        w = (lx*lz).sum()
        overlap = max(overlap, w)
    assert overlap <= 1, overlap
    #print("max overlap:", overlap)

    assert rank(Hx) == len(Hx)
    assert rank(Hz) == len(Hz)
    mx = len(Hx)
    mz = len(Hz)

    # ---------------------------------------------------
    # Remove excess logops.

    print("remove_dependent")

    # -------- Lx

    Lx, Lx1 = mk_logops(Lx, Hx)

    # -------- Lz

    Lz, Lz1 = mk_logops(Lz, Hz)

    #LzHz = numpy.concatenate((Lz, Hz), axis=0)
    #LzHz = remove_dependent(LzHz)
    #assert rank(LzHz) == len(LzHz)
    #assert len(LzHz) >= mz
    #assert eq2(LzHz[-len(Hz):], Hz)
    #Lz = LzHz[:-mz]

    # ---------------------------------------------------
    # 

    k = len(Lx)
    assert k == len(Lz)

    #assert eq2(dot2(Lz, Lxt), identity2(k))
    assert mx + mz + k == n

    print("mx = %d, mz = %d, k = %d" % (mx, mz, k))

    # ---------------------------------------------------
    # 

    op = zeros2(n)
    for lx in Lx:
      for lz in Lz:
        lxz = lx*lz
        #print(lxz)
        #print(op.shape, lxz.shape)
        op += lxz

    for lx in Lx1:
      for lz in Lz1:
        lxz = lx*lz
        #print(lxz)
        #print(op.shape, lxz.shape)
        op += lxz

    idxs = numpy.where(op)[0]
    print("correctable region size = %d" % len(idxs))
    #print(op)
    #print(idxs)

    good = is_correctable(n, idxs, Lx, Lz)
    assert good

    # ---------------------------------------------------
    # 

    if argv.code:
        print("code = CSSCode()")
        code = CSSCode(Hx=Hx, Hz=Hz, Lx=Lx, Lz=Lz, check=True, verbose=False, build=True)
        print(code)
        #print(code.weightstr())
    
        if check:
            U = solve(Lx.transpose(), code.Lx.transpose())
            assert U is not None
            #print(U.shape)
            assert eq2(dot2(U.transpose(), Lx), code.Lx)
            #print(shortstr(U))
    
        if 0:
            Lx, Lz = code.Lx, code.Lz
            print("Lx:", Lx.shape)
            print(shortstr(Lx))
            print("Lz:", Lz.shape)
            print(shortstr(Lz))




def main():

    if argv.ldpc:
        # LDPC
        l = argv.get("l", 3) # column weight
        m = argv.get("m", 4) # row weight
        n = argv.get("n", 8) # cols
        r = argv.get("r", n*l//m) # rows
        d = argv.get("d", 1) # distance
        C = make_gallagher(r, n, l, m, d)
        #print(shortstr(C))
        if argv.same:
            D = C
        else:
            D = make_gallagher(r, n, l, m, d)
            assert rank(C) == len(C)
            assert rank(D) == len(D)
        print("rank(C)", rank(C), "kernel(C)", len(find_kernel(C)))
        print("rank(D)", rank(D), "kernel(D)", len(find_kernel(D)))

    elif argv.torus:
        # Torus
        C = parse("""
        11..
        .11.
        ..11
        1..1
        """)
        D = C
    elif argv.hamming:
        C = parse("""
        ...1111
        .11..11
        1.1.1.1
        """)
        D = C
    else:
        # Surface
        C = parse("""
        11..
        .11.
        ..11
        """)
        D = C

    Ct = C.transpose()
    Dt = D.transpose()
    hypergraph_product(C, Dt)
    #hypergraph_product(D, D)
    #hypergraph_product(C, C)
    #hypergraph_product(D, C)


def latex(A):
    rows, cols = A.shape
    lines = []
    lines.append(r"\begin{array}{%s}" % ('c'*cols,))
    for row in A:
        line = [str(x) for x in row]
        line = ' & '.join(line) + r"\\"
        lines.append(line)
    lines.append(r"\end{array}")
    s = '\n'.join(lines)
    s = s.replace(" 0 ", " . ")
    return s
    


def test():
    H = parse("""
     1001011
     0101110
     0010111
    """)
    print(latex(H))
    G = parse("""
     1101000
     0110100
     1110010
     1010001
    """)
    print(latex(G))
    print(dot2(G, H.transpose()))


if __name__ == "__main__":

    _seed = argv.get("seed")
    if _seed is not None:
        seed(_seed)
        ra.seed(_seed)

    while 1:
        main()
        #test()
        if not argv.forever:
            break





