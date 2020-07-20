#!/usr/bin/env python3

"""
from previous version: transverse.py

"""

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

    def h_mark(self, row, col, st_bar=None, st_qubit=None, 
            r=None, stroke=False, fill=False):
        (c0, c1, d0, d1) = self.shape
        assert 0<=row<c1
        assert 0<=col<d0
        st_qubit = st_qubit or self.st_qubit
        st_bar = st_bar or self.st_bar
        dx, dy, r = self.dx, self.dy, (r or self.r)
        cvs = self.cvs
        x, y = dx*col, dy*row
        cvs.stroke(path.line(x-0.4*dx, y, x+0.4*dx, y), st_bar)
        if fill:
            cvs.fill(path.circle(x, y, r), st_qubit)
        if stroke:
            cvs.stroke(path.circle(x, y, r), st_qubit)

    def v_mark(self, row, col, st_bar=None, st_qubit=None, 
            r=None, stroke=False, fill=False):
        (c0, c1, d0, d1) = self.shape
        assert 0<=row<c0
        assert 0<=col<d1
        st_qubit = st_qubit or self.st_qubit
        st_bar = st_bar or self.st_bar
        dx, dy, r = self.dx, self.dy, (r or self.r)
        cvs = self.cvs
        x, y = dx*(col+0.5), dy*(row+0.5)
        cvs.stroke(path.line(x, y-0.4*dy, x, y+0.4*dy), st_bar)
        if fill:
            cvs.fill(path.circle(x, y, r), st_qubit)
        if stroke:
            cvs.stroke(path.circle(x, y, r), st_qubit)

    def get_hidx(self, row, col):
        (c0, c1, d0, d1) = self.shape
        idx = row*c1 + col
        assert 0 <= idx < c1*d0
        return idx

    def get_vidx(self, row, col):
        (c0, c1, d0, d1) = self.shape
        idx = c1*d0 + col*c0 + row
        assert c1*d0 <= idx < c1*d0 + c0*d1
        return idx

    def mark(self, i, *args, **kw):
        (c0, c1, d0, d1) = self.shape
        #print("mark", i)
        if i < c1*d0:
            # horizontal
            col = i%d0
            row = i//c1
            assert i == col + row*c1
            self.h_mark(row, col, *args, **kw)
        else:
            # vertical ... may be row/col transposed ...
            i -= c1*d0
            assert i < c0*d1
            row = i%d1
            col = i//c0
            assert i == row + col*c0 # ?
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


def dependent_rows(H):
    "find dependent rows of H, first to last"
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

    KerC = find_kernel(C)
    #KerC = min_span(KerC) # does not seem to matter... ??
    assert KerC.shape[1] == c1
    K = KerC.transpose()
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

    # These are linearly dependent, but 
    # once we add stabilizers it will be reduced:
    assert rank(Lx) == len(Lx)


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
    # 

    while 1:
    
        Lxi = independent_logops(Lx, Hx)
        Lzi = independent_logops(Lz, Hz)
    
        k = len(Lxi)
        assert len(Lzi) == k
        assert mx + mz + k == n
    
        LxiHx = numpy.concatenate((Lxi, Hx))
        assert rank(LxiHx) == k+mx
        LziHz = numpy.concatenate((Lzi, Hz))
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
    
        #Ax = in_support(LxiHx, idxs)
        #print(Ax.shape)
    
        assert dot2(LxiHx, Hz.transpose()).sum() == 0
    
        A = identity2(n)[idxs]
        Ax = intersect(LxiHx, A)
        Az = intersect(LziHz, A)
    
        assert dot2(Ax, Hz.transpose()).sum() == 0
        assert dot2(Az, Hx.transpose()).sum() == 0
    
        print("Ax:")
        print(shortstr(Ax))
        print("Az:")
        print(shortstr(Az))
    
        for z in Az:
            draw.mark_zop(z)
            break
    
        if dot2(Ax, Lz.transpose()).sum() == 0 and dot2(Az, Lx.transpose()).sum() == 0:
            break

        draw.save("output.2")

        Lx = shuff2(Lx)
        Lz = shuff2(Lz)
    

def shuff2(A):
    idxs = list(range(len(A)))
    shuffle(idxs)
    A = A[idxs]
    return A
    

def main():

    if argv.ldpc:
        # LDPC
        l = argv.get("l", 3) # column weight
        m = argv.get("m", 4) # row weight
        n = argv.get("n", 8) # cols
        r = argv.get("r", n*l//m) # rows
        d = argv.get("d", 1) # distance
        C = make_gallagher(r, n, l, m, d)
        print(shortstr(C))
        print("rank(C)", rank(C), "kernel(C)", len(find_kernel(C)))
        if argv.same:
            D = C
        else:
            D = make_gallagher(r, n, l, m, d)
            assert rank(C) == len(C)
            assert rank(D) == len(D)
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
    elif argv.surf or argv.surface:
        # Surface
        C = parse("""
        11..
        .11.
        ..11
        """)
        D = C
    else:
        return

    Ct = C.transpose()
    Dt = D.transpose()
    hypergraph_product(C, Dt)



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





