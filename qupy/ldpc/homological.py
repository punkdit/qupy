#!/usr/bin/env python3

"""
from previous version: transverse.py

see also: classical.py

"""

import sys
from random import randint, seed, random, shuffle

import numpy
import numpy.random as ra
from numpy.linalg import lstsq

from qupy.argv import argv
from qupy.tool import write, choose
from qupy.ldpc.css import CSSCode, randcss
from qupy.ldpc.solve import shortstr, zeros2, array2, dot2, parse, identity2, rank
from qupy.ldpc.solve import find_kernel, cokernel, eq2, get_reductor, intersect
from qupy.ldpc.solve import linear_independent, solve, row_reduce, rand2
from qupy.ldpc.solve import remove_dependent, dependent_rows
from qupy.ldpc.gallagher import make_gallagher, classical_distance


from huygens.front import *
from huygens.box import *

red = color.rgb(0.7, 0.2, 0.2)
green = color.rgb(0.2, 0.7, 0.2)
blue = color.rgb(0.2, 0.2, 0.7)

st_thick = [style.linewidth.Thick]


class Draw(object):
    def __init__(self, c0, c1, d0, d1):
        cvs = canvas.canvas()
        self.dx = 0.4
        self.dy = self.dx
        self.r = 0.10

        self.st_qubit = [color.rgb.grey]
        self.st_bar = [color.rgb.black]

        self.cvs = cvs
        self.shape = (c0, c1, d0, d1)

        # horizontal edges --------
        for row in range(c1):
          for col in range(d0):
            self.h_mark(row, col, fill=True)

        # vertical edges --------
        for row in range(c0):
          for col in range(d1):
            self.v_mark(row, col, fill=True)

        print(self)

        rows = max(c0, c1)
        cols = max(d0, d1)
        dx, dy = self.dx, self.dy
        p = path.rect(-dx, -dy, cols*dy+2*dy, rows*dx+dx)
        p = path.rect(-0.5*dx, -0.5*dy, cols*dy+0.0*dy, rows*dx+0.0*dx)
        cvs.stroke(p, [color.rgb.grey])

    def __str__(self):
        return "Draw%s"%((self.shape,))

    def h_mark(self, row, col, st_bar=None, st_qubit=None, 
            r=None, stroke=False, fill=False):
        (c0, c1, d0, d1) = self.shape
        #assert 0<=row<c1
        #assert 0<=col<d0
        st_qubit = st_qubit or self.st_qubit
        st_bar = st_bar or self.st_bar
        dx, dy, r = self.dx, self.dy, (r or self.r)
        cvs = self.cvs
        x, y = dx*col, dy*row
        if fill:
            cvs.fill(path.circle(x, y, r), st_qubit)
        if stroke:
            cvs.stroke(path.circle(x, y, r), st_qubit)
        cvs.stroke(path.line(x-0.4*dx, y, x+0.4*dx, y), st_bar)

    def v_mark(self, row, col, st_bar=None, st_qubit=None, 
            r=None, stroke=False, fill=False):
        (c0, c1, d0, d1) = self.shape
        #assert 0<=row<c0
        #assert 0<=col<d1
        st_qubit = st_qubit or self.st_qubit
        st_bar = st_bar or self.st_bar
        dx, dy, r = self.dx, self.dy, (r or self.r)
        cvs = self.cvs
        x, y = dx*(col+0.5), dy*(row+0.5)
        if fill:
            cvs.fill(path.circle(x, y, r), st_qubit)
        if stroke:
            cvs.stroke(path.circle(x, y, r), st_qubit)
        cvs.stroke(path.line(x, y-0.4*dy, x, y+0.4*dy), st_bar)

    def get_hidx(self, row, col):
        (c0, c1, d0, d1) = self.shape
        idx = row*d0 + col
        assert 0 <= idx < c1*d0
        return idx

    def get_vidx(self, row, col):
        (c0, c1, d0, d1) = self.shape
        idx = c1*d0 + col*c0 + row # ???
        assert c1*d0 <= idx < c1*d0 + c0*d1
        return idx

    def mark(self, i, *args, **kw):
        (c0, c1, d0, d1) = self.shape
        #print("mark", i)
        if i < c1*d0:
            # horizontal
            col = i%d0
            row = i//d0
            assert i == col + row*d0
            self.h_mark(row, col, *args, **kw)
        else:
            # vertical ... may be row/col transposed ...
            i -= c1*d0
            assert i < c0*d1
            col = i%d1
            row = i//d1
            assert i == col + row*d1
            self.v_mark(row, col, *args, **kw)

    def mark_op(self, op, *args, **kw):
        (c0, c1, d0, d1) = self.shape
        n = len(op)
        assert n == c1*d0 + c0*d1
        for i in range(n):
            if op[i]:
                self.mark(i, *args, **kw)

    def mark_xop(self, op):
        self.mark_op(op, st_qubit=[blue]+st_thick, r=0.06, stroke=True)

    def mark_zop(self, op):
        self.mark_op(op, st_qubit=[green]+st_thick, r=0.12, stroke=True)

    def mark_idx(self, idx):
        self.mark(idx, st_qubit=[red]+st_thick, r=0.16, stroke=True)

    def save(self, name):
        self.cvs.writePDFfile(name+".pdf")
        self.cvs.writeSVGfile(name+".svg")


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


#def dependent_rows(H):
#    "find dependent rows of H, first to last"
#    idxs = set(range(len(H)))
#    #print(H)
#    K = find_kernel(H.transpose())
#    #print("K:")
#    #print(K)
#    K = row_reduce(K, truncate=True)
#    #print("K:")
#    #print(K)
#    assert dot2(K, H).sum() == 0
#    deps = []
#    for row in K:
#        #print(row)
#        idx = numpy.where(row!=0)[0][0]
#        deps.append(idx)
#        idxs.remove(idx)
#    assert len(set(deps)) == len(K)
#    idxs = list(idxs)
#    idxs.sort()
#    deps.sort()
#    return idxs, deps
#
#
#def remove_dependent(H):
#    "remove dependent rows of H, first to last"
#    if len(H) <2:
#        return H
#    idxs, deps = dependent_rows(H)
#    return H[idxs]



def independent_logops(L, H):
    m = len(H)
    LH = numpy.concatenate((L, H), axis=0)
    keep, remove = dependent_rows(LH)
    LH = LH[keep]
    
    assert rank(LH) == len(LH)
    assert len(LH) >= m
    assert eq2(LH[-len(H):], H)
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


def rand_span(A):
    while 1:
        m, n = A.shape
        v = rand2(m, m)
        A1 = dot2(v, A)
        assert A1.shape == A.shape
        if rank(A) == rank(A1):
            break
    assert rank(intersect(A, A1)) == rank(A)
    return A1


def hypergraph_product(C, D, check=False):
    print("hypergraph_product: C=%s, D=%s"%(C.shape, D.shape))
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

    KerC = find_kernel(C)
    #KerC = min_span(KerC) # does not seem to matter... ??
    KerC = rand_span(KerC) # ??
    KerC = row_reduce(KerC)

    assert KerC.shape[1] == c1
    K = KerC.transpose()
    #K = min_span(K)
    #K = rand_span(K)
    E = identity2(d0)

    #print(shortstr(KerC))
    #print()

    Lxt0 = kron(K, E), zeros2(c0*d1, K.shape[1]*d0)
    Lxt0 = numpy.concatenate(Lxt0, axis=0)
    assert dot2(Hz, Lxt0).sum() == 0

    K = find_kernel(D).transpose()
    assert K.shape[0] == d1
    E = identity2(c0)

    Lxt1 = zeros2(c1*d0, K.shape[1]*c0), kron(E, K)
    Lxt1 = numpy.concatenate(Lxt1, axis=0)
    assert dot2(Hz, Lxt1).sum() == 0

    Lxt = numpy.concatenate((Lxt0, Lxt1), axis=1) # horizontal concatenate
    Lx = Lxt.transpose()

    assert dot2(Hz, Lxt).sum() == 0

    # These are linearly independent among themselves, but 
    # once we add stabilizers it will be reduced:
    assert rank(Lx) == len(Lx)


    # ---------------------------------------------------
    # Build Lz 

    counit = lambda n : unit2(n).transpose()

    CokerD = find_cokernel(D) # matrix of row vectors
    #print("CokerD")
    #print(CokerD)
    #CokerD = min_span(CokerD)
    CokerD = rand_span(CokerD)
    #print("CokerD")
    #print(CokerD)
    #print(shortstr(CokerD))

    #E = counit(c1)
    E = identity2(c1)
    Lz0 = kron(E, CokerD), zeros2(CokerD.shape[0]*c1, c0*d1)
    Lz0 = numpy.concatenate(Lz0, axis=1) # horizontal concatenate

    assert dot2(Lz0, Hx.transpose()).sum() == 0

    K = find_cokernel(C)
    #E = counit(d1)
    E = identity2(d1)
    Lz1 = zeros2(K.shape[0]*d1, c1*d0), kron(K, E)
    Lz1 = numpy.concatenate(Lz1, axis=1) # horizontal concatenate

    Lz = numpy.concatenate((Lz0, Lz1), axis=0)

    assert dot2(Lz, Hx.transpose()).sum() == 0

    #print(shortstr(Lz))

    # ---------------------------------------------------
    # 

    overlap = 0
    for lx in Lx:
      for lz in Lz:
        w = (lx*lz).sum()
        overlap = max(overlap, w)
    assert overlap <= 1, overlap
    #print("max overlap:", overlap)

    if 0:
        # here we assume that Hx/Hz are full rank
        Hxi = Hx
        Hzi = Hz
        assert rank(Hx) == len(Hx)
        assert rank(Hz) == len(Hz)
        mx = len(Hx)
        mz = len(Hz)
    else:
        Hxi = remove_dependent(Hx)
        Hzi = remove_dependent(Hz)
        mx = rank(Hx)
        mz = rank(Hz)
        assert len(Hxi) == mx
        assert len(Hzi) == mz


    # ---------------------------------------------------
    # 

    # This makes len(idxs) much bigger, because we
    # end up with logops from many different rows/cols:
    #Lx = shuff2(Lx)
    #Lz = shuff2(Lz)

    Lxi = independent_logops(Lx, Hxi)
    Lzi = independent_logops(Lz, Hzi)

    print("Lxi:", len(Lxi))
    print("Lzi:", len(Lzi))

    k = len(Lxi)
    assert len(Lzi) == k
    assert mx + mz + k == n

    LxiHx = numpy.concatenate((Lxi, Hxi))
    assert rank(LxiHx) == k+mx
    LziHz = numpy.concatenate((Lzi, Hzi))
    assert rank(LziHz) == k+mz

    op = zeros2(n)
    for lx in Lxi:
      for lz in Lzi:
        lxz = lx*lz
        #print(lxz)
        #print(op.shape, lxz.shape)
        op += lxz

    idxs = numpy.where(op)[0]

    draw = Draw(c0, c1, d0, d1)
    for idx in idxs:
        draw.mark_idx(idx)

    draw.mark_xop(Lx[0])
    print(KerC)
    for j, op in enumerate(KerC):
      for i in range(c1):
        row = i
        col = -j-1
        if op[i]:
            draw.h_mark(row, col, st_qubit=[blue], stroke=True)
        else:
            draw.h_mark(row, col)

    print(CokerD.shape)
    print(CokerD)
    draw.mark_zop(Lz[0])
    for j, op in enumerate(CokerD):
      print(op)
      for i in range(d0):
        col = i
        row = -j-1
        if op[i]:
            draw.h_mark(row, col, st_qubit=[green], stroke=True)
        else:
            draw.h_mark(row, col)


    draw.save("output.2")

    print("len(idxs) =", len(idxs))

    #Ax = in_support(LxiHx, idxs)
    #print(Ax.shape)

    assert dot2(LxiHx, Hz.transpose()).sum() == 0

    A = identity2(n)[idxs]
    Ax = intersect(LxiHx, A)
    Az = intersect(LziHz, A)

    assert dot2(Ax, Hz.transpose()).sum() == 0
    assert dot2(Az, Hx.transpose()).sum() == 0

    if dot2(Ax, Lz.transpose()).sum() == 0 and dot2(Az, Lx.transpose()).sum() == 0:
        return True

    #draw.mark_zop(Az[0])
    print("Ax:")
    print(shortstr(Ax))
    print("Az:")
    print(shortstr(Az))

    return False


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
    

def random_code(n, _rank, deps, distance=1):
    d = 0
    while d<distance:
        H = rand2(_rank, n)
        if rank(H) < _rank:
            continue
        d = classical_distance(H, distance)

    while deps>0:
        R = rand2(deps, _rank)
        J = dot2(R, H)
        K = numpy.concatenate((H, J))
        if rank(K) == _rank:
            H = K
            break

    return H


def main():

    if argv.ldpc:
        # LDPC
        l = argv.get("l", 3) # column weight
        m = argv.get("m", 4) # row weight
        n = argv.get("n", 8) # cols
        r = argv.get("r", n*l//m) # rows
        d = argv.get("d", 1) # distance
        print("make_gallagher%s"%((r, n, l, m, d),))
        C = make_gallagher(r, n, l, m, d)
        print(shortstr(C))
        print()
        print(shortstr(C))
        print("rank(C) = ", rank(C), "kernel(C) = ", len(find_kernel(C)))
        if argv.same:
            D = C
        else:
            D = make_gallagher(r, n, l, m, d)
            assert rank(C) == len(C)
            assert rank(D) == len(D)
            print("rank(D)", rank(D), "kernel(D)", len(find_kernel(D)))

    elif argv.rand:
        # make some vertical logops from rank degenerate parity check matrices
        C = random_code(20, 15, 5, 3)
        D = random_code(8, 6, 2, 3)
        #dC = classical_distance(C)
        #dD = classical_distance(D)
        print("C", C.shape, rank(C))
        print(shortstr(C))
        print("D", D.shape, rank(D))
        print(shortstr(D))

    elif argv.pair:
        #C = make_gallagher(9, 12, 3, 4, 4) # big
        C = make_gallagher(15, 20, 3, 4, 4) # big
        D = make_gallagher(6, 8, 3, 4, 1) # small

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
        111....
        """)
        D = C

    elif argv.surf or argv.surface:
        # Surface
        C = parse("""
        11....
        .11...
        ..11..
        ...11.
        ....11
        """)
        D = parse("""
        11..
        .11.
        ..11
        """)

    elif argv.small:
        C = parse("""1111""")
        D = parse("""1111""")

    else:
        return

    Ct = C.transpose()
    Dt = D.transpose()

    if argv.dual:
        C, Ct = Ct, C
        D, Dt = Dt, D

    while 1:
        success = hypergraph_product(C, Dt)
        print("success:", success)
        if success:
            break
        #else:
        #    sys.exit(0)
        C = shuff22(C)
        if argv.same:
            D = C
            Dt = D.transpose()
        else:
            Dt = shuff22(Dt)



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





