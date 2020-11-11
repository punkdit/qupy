#!/usr/bin/env python3

"""
from previous version: homological.py (and transverse.py)

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
from qupy.ldpc.solve import shortstrx, zeros2, array2, dot2, parse, identity2, rank
from qupy.ldpc.solve import find_kernel, cokernel, eq2, get_reductor, intersect
from qupy.ldpc.solve import linear_independent, solve, row_reduce, rand2
from qupy.ldpc.solve import remove_dependent, dependent_rows
from qupy.ldpc.gallagher import make_gallagher, classical_distance
from qupy.ldpc.main import get_decoder

shortstr = shortstrx


from huygens.front import *
from huygens.box import *

red = color.rgb(0.7, 0.2, 0.2)
green = color.rgb(0.2, 0.7, 0.2)
blue = color.rgb(0.2, 0.2, 0.7)

st_thick = [style.linewidth.Thick]


class Draw(object):
    def __init__(self, c0, c1, d0, d1, hspace=7):
        cvs = canvas.canvas()
        self.dx = 0.4
        self.dy = self.dx
        self.r = 0.10

        self.st_qubit = [color.rgb.grey]
        self.st_bar = [color.rgb.black]

        self.cvs = cvs
        self.shape = (c0, c1, d0, d1)
        self.hspace = hspace

        #print(self)

        rows, cols = c1, d0
        dx, dy = self.dx, self.dy
        p = path.rect(-0.5*dx, -0.5*dy, cols*dy+0.0*dy, rows*dx+0.0*dx)
        cvs.fill(p, [color.rgb.grey])
        cvs.stroke(p, [color.rgb.black])

        rows, cols = c0, d1
        space = d0 + self.hspace
        p = path.rect(-0.5*dx + space*dx, -0.5*dy, cols*dy+0.0*dy, rows*dx+0.0*dx)
        cvs.fill(p, [color.rgb.grey])
        cvs.stroke(p, [color.rgb.black])

        # horizontal edges --------
        for row in range(c1):
          for col in range(d0):
            self.h_mark(row, col, fill=True)

        # vertical edges --------
        for row in range(c0):
          for col in range(d1):
            self.v_mark(row, col, fill=True)

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
        if 0:
            x, y = dx*(col+0.5), dy*(row+0.5)
        else:
            row1 = row # - c0
            col1 = col + d0 + self.hspace
            x, y = dx*(col1+0.0), dy*(row1+0.0)
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

    c0, c1 = C.shape
    d0, d1 = D.shape

    Ic0 = identity2(c0)
    Id0 = identity2(d0)
    Ic1 = identity2(c1)
    Id1 = identity2(d1)

    Hz0 = kron(Ic1, D.transpose()), kron(C.transpose(), Id1)
    Hz = numpy.concatenate(Hz0, axis=1) # horizontal concatenate

    Hx0 = kron(C, Id0), kron(Ic0, D)
    #print("Hx0:", Hx0[0].shape, Hx0[1].shape)
    Hx = numpy.concatenate(Hx0, axis=1) # horizontal concatenate

    assert dot2(Hx, Hz.transpose()).sum() == 0

    n = Hz.shape[1]
    assert Hx.shape[1] == n

    # ---------------------------------------------------
    # Build Lz 

    KerC = find_kernel(C)
    #KerC = min_span(KerC) # does not seem to matter... ??
    KerC = rand_span(KerC) # ??
    KerC = row_reduce(KerC)

    assert KerC.shape[1] == c1
    K = KerC.transpose()
    #K = min_span(K)
    #K = rand_span(K)
    #E = identity2(d0)

    #print("c0,c1,d0,d1=", c0, c1, d0, d1)

    Lzt0 = kron(K, Id0), zeros2(c0*d1, K.shape[1]*d0)
    Lzt0 = numpy.concatenate(Lzt0, axis=0)
    assert dot2(Hx, Lzt0).sum() == 0

    KerD = find_kernel(D)
    K = KerD.transpose()
    assert K.shape[0] == d1

    Lzt1 = zeros2(c1*d0, K.shape[1]*c0), kron(Ic0, K)
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

    CokerD = find_cokernel(D) # matrix of row vectors
    #CokerD = min_span(CokerD)
    CokerD = rand_span(CokerD)

    Lx0 = kron(Ic1, CokerD), zeros2(CokerD.shape[0]*c1, c0*d1)
    Lx0 = numpy.concatenate(Lx0, axis=1) # horizontal concatenate

    assert dot2(Lx0, Hz.transpose()).sum() == 0

    CokerC = find_cokernel(C)
    Lx1 = zeros2(CokerC.shape[0]*d1, c1*d0), kron(CokerC, Id1)
    Lx1 = numpy.concatenate(Lx1, axis=1) # horizontal concatenate

    Lx = numpy.concatenate((Lx0, Lx1), axis=0)

    assert dot2(Lx, Hz.transpose()).sum() == 0

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


def test_indep(Lx0, Lx1, Hxi, Hx, **kw):

    if argv.verbose:
        print("Lx0:", Lx0.shape)
        print(shortstr(Lx0))
        print("Lx1:", Lx1.shape)
        print(shortstr(Lx1))
        print("Hxi:", Hxi.shape)
        print(shortstr(Hxi))

    Lx0 = independent_logops(Lx0, Hxi)
    Lx1 = independent_logops(Lx1, Hxi)

    i0, i1, i2 = len(Lx0), len(Lx1), len(Hxi)
    J = numpy.concatenate((Lx0, Lx1, Hxi))
    K = find_cokernel(J)
    print("K:", K.shape)
    print(shortstrx(K[:,:i0], K[:, i0:i0+i1], K[:, i0+i1:i0+i1+i2]))
    assert dot2(K, J).sum() == 0


def test_code(Hxi, Hzi, Hx, Lx, Lz, Lx0, Lx1, LxiHx, **kw):
    code = CSSCode(Hx=Hxi, Hz=Hzi)
    print(code)

    assert rank(intersect(Lx, code.Lx)) == code.k
    assert rank(intersect(Lz, code.Lz)) == code.k

    verbose = argv.verbose
    decoder = get_decoder(argv, argv.decode, code)
    
    if decoder is None:
        return

    p = argv.get("p", 0.01)
    N = argv.get("N", 0)

    distance = code.n
    count = 0
    failcount = 0
    nonuniq = 0

    logops = []

    for i in range(N):
        err_op = ra.binomial(1, p, (code.n,))
        err_op = err_op.astype(numpy.int32)
        op = decoder.decode(p, err_op, verbose=verbose, argv=argv)
    
        c = 'F'
        success = False
        if op is not None:
            op = (op+err_op)%2
            # Should be a codeword of Hz (kernel of Hz)
            assert dot2(code.Hz, op).sum() == 0
            write("%d:"%op.sum())
        
            # Are we in the image of Hx ? If so, then success.
            success = dot2(code.Lz, op).sum()==0
        
            if success and op.sum():
                nonuniq += 1
        
            c = '.' if success else 'x'

            if op.sum() and not success:
                distance = min(distance, op.sum())
                write("L")
                logops.append(op.copy())

        else:
            failcount += 1
        write(c+' ')
        count += success

    if N:
        print()
        print(argv)
        print("error rate = %.8f" % (1. - 1.*count / (i+1)))
        print("fail rate = %.8f" % (1.*failcount / (i+1)))
        print("nonuniq = %d" % nonuniq)
        print("distance <= %d" % distance)

    mx0, mx1 = len(Lx0), len(Lx1)
    LxHx = numpy.concatenate((Lx0, Lx1, Hx))
    for op in logops:
        print(op.sum())
        #print(shortstr(op))
        #print(op.shape)
        #print(op)
        K = solve(LxHx.transpose(), op)
        K.shape = (1, len(K))
        print(shortstrx(K[:, :mx0], K[:, mx0:mx0+mx1], K[:, mx0+mx1:]))


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


def get_bipuncture(H, G=None, verbose=True):
    n = H.shape[1]
    if G is None:
        G = find_kernel(H)
    k = len(G)

    if verbose:
        print(shortstrx(G, H))

    while 1: # copied from classical.py
        idxs = set()
        while len(idxs) < k:
            idx = randint(0, n-1)
            idxs.add(idx)
        idxs = list(idxs)
        idxs.sort()

        G1 = in_support(G, idxs)
        if len(G1):
            continue

        jdxs = set()
        while len(jdxs) < k:
            jdx = randint(0, n-1)
            if jdx not in idxs:
                jdxs.add(jdx)
        jdxs = list(jdxs)
        jdxs.sort()

        G2 = in_support(G, jdxs)
        if len(G2) == 0:
            break
    
    if 0:
        v = zeros2(1, n)
        v[:, idxs] = 1
        print(shortstr(v))
    
        v = zeros2(1, n)
        v[:, jdxs] = 1
        print(shortstr(v))

    if verbose:
        print("in_support(H):", in_support(H, idxs), in_support(H, jdxs))

    return idxs, jdxs


def is_correctable(idxs, n, LxiHx, LziHz, Hx, Hz, Lx, Lz, **kw):
    #print("len(idxs) =", len(idxs))
    #Ax = in_support(LxiHx, idxs)
    #print(Ax.shape)

    A = identity2(n)[idxs]
    Ax = intersect(LxiHx, A)
    Az = intersect(LziHz, A)

    assert dot2(Ax, Hz.transpose()).sum() == 0
    assert dot2(Az, Hx.transpose()).sum() == 0

    if dot2(Ax, Lz.transpose()).sum() == 0 and dot2(Az, Lx.transpose()).sum() == 0:
        return True

    if 0:
        #draw.mark_zop(Az[0])
        print("Ax:")
        print(shortstr(Ax))
        print("Az:")
        print(shortstr(Az))

    return False


def eq_span(A, B):
    u = solve(A.transpose(), B.transpose())
    if u is None:
        return False
    u = solve(B.transpose(), A.transpose())
    return u is not None


def test_overlap(n, mx, mz, k, c0, c1, d0, d1, 
        Hx, Hz, Lx, Lz, Lxi, Lzi, LxiHx, LziHz, 
        Ic0, Ic1, Id0, Id1,
        C, D, KerC, CokerD, KerD, CokerC, **kw):

    if 0:
        # This makes len(idxs) much bigger, because we
        # end up with logops from many different rows/cols:
        Lx = shuff2(Lx)
        Lz = shuff2(Lz)

    assert dot2(LxiHx, Hz.transpose()).sum() == 0

    def get_logops(idxs_C, idxs_Ct, idxs_D, idxs_Dt):
    
        # Lx --------------------------------------------------------------
    
        Ic1 = identity2(c1)[idxs_C, :]
        Lx_h = kron(Ic1, CokerD), zeros2(len(idxs_C)*CokerD.shape[0], c0*d1)
        Lx_h = numpy.concatenate(Lx_h, axis=1)
        assert dot2(Lx_h, Hz.transpose()).sum() == 0
    
        Id1 = identity2(d1)[idxs_D, :]
        Lx_v = zeros2(CokerC.shape[0]*len(idxs_D), c1*d0), kron(CokerC, Id1)
        Lx_v = numpy.concatenate(Lx_v, axis=1)
        Lxi = numpy.concatenate((Lx_h, Lx_v), axis=0)
    
        # Lz --------------------------------------------------------------
    
        KerCt = KerC.transpose()
        Id0 = identity2(d0)[:, idxs_Dt]
        Lzt_h = kron(KerCt, Id0), zeros2(c0*d1, KerCt.shape[1]*len(idxs_Dt))
        Lzt_h = numpy.concatenate(Lzt_h, axis=0)
        assert dot2(Hx, Lzt_h).sum() == 0
    
        KerDt = KerD.transpose()
        assert KerDt.shape[0] == d1
        Ic0 = identity2(c0)[:, idxs_Ct]
        Lzt_v = zeros2(c1*d0, len(idxs_Ct)*KerDt.shape[1]), kron(Ic0, KerDt)
        Lzt_v = numpy.concatenate(Lzt_v, axis=0)
        assert dot2(Hx, Lzt_v).sum() == 0
    
        Lzti = numpy.concatenate((Lzt_h, Lzt_v), axis=1)
        Lzi = Lzti.transpose()

        # checking ---------------------------------------------------------

        assert dot2(Hx, Lzti).sum() == 0

        assert rank(Lxi) == len(Lxi) # full rank
        assert rank(Lzi) == len(Lzi) # full rank

        assert eq_span(numpy.concatenate((Lxi, Hx)), LxiHx)
        assert eq_span(numpy.concatenate((Lzi, Hz)), LziHz)

        return Lxi, Lzi


    print("get_bipuncture(C)")
    l_C,  r_C  = get_bipuncture(C)
    print("get_bipuncture(Ct)")
    l_Ct, r_Ct = get_bipuncture(C.transpose())
    print("get_bipuncture(D)")
    l_D,  r_D  = get_bipuncture(D)
    print("get_bipuncture(Dt)")
    l_Dt, r_Dt = get_bipuncture(D.transpose())
    print("done.")

    left_Lxi, left_Lzi = get_logops(l_C, l_Ct, l_D, l_Dt)
    right_Lxi, right_Lzi = get_logops(r_C, r_Ct, r_D, r_Dt)

    randvec = lambda U : dot2(rand2(1, len(U)), U)

    print("shapes:", left_Lxi.shape, right_Lzi.shape, left_Lzi.shape, right_Lxi.shape)
    #while 1:
    for trial in range(100):
        l_op = randvec(left_Lxi)*randvec(right_Lzi) # intersection
        r_op = randvec(right_Lxi)*randvec(left_Lzi) # intersection
        assert (l_op * r_op).sum() == 0
        op = l_op + r_op # union
        idxs = numpy.where(op)[0]
        print("(%.2f)"%(len(idxs)/n), end="", flush=True)
        assert is_correctable(**locals())
    print()

    l_op = zeros2(n)
    for lx in left_Lxi:
      for lz in right_Lzi:
        lxz = lx*lz
        l_op += lxz

    r_op = zeros2(n)
    for lx in right_Lxi:
      for lz in left_Lzi:
        lxz = lx*lz
        r_op += lxz

    assert (l_op * r_op).sum() == 0
    op = l_op + r_op
    #op = l_op

    idxs = numpy.where(op)[0]
    print("--> (%.2f)\n"%(len(idxs)/n), end="", flush=True)

    #assert is_correctable(**locals())

    draw = Draw(c0, c1, d0, d1)
    for idx in idxs:
        draw.mark_idx(idx)

    if 0:
        #draw.mark_xop(Lx[0])
        for xop in Lxi:
            draw.mark_xop(xop)
    
        #draw.mark_zop(Lz[0])
        for zop in Lzi:
            draw.mark_zop(zop)

    # h_mark
    assert KerC.shape[1] == c1
    for j, op in enumerate(KerC):
      for i in range(c1):
        row = i
        col = -j-1
        if op[i]:
            draw.h_mark(row, col, st_qubit=[green], stroke=True)
        else:
            draw.h_mark(row, col)

    assert CokerD.shape[1] == d0
    for j, op in enumerate(CokerD):
      #print(op)
      for i in range(d0):
        col = i
        #row = -j-1
        row = c1+j
        if op[i]:
            draw.h_mark(row, col, st_qubit=[blue], stroke=True)
        else:
            draw.h_mark(row, col)

    # v_mark
    assert KerD.shape[1] == d1
    for j, op in enumerate(KerD):
      for i in range(d1):
        col = i
        #row = -j-1
        row = c0+j
        if op[i]:
            draw.v_mark(row, col, st_qubit=[green], stroke=True)
        else:
            draw.v_mark(row, col)

    assert CokerC.shape[1] == c0
    for j, op in enumerate(CokerC):
      #print(op)
      for i in range(c0):
        row = i
        col = -j-1
        if op[i]:
            draw.v_mark(row, col, st_qubit=[blue], stroke=True)
        else:
            draw.v_mark(row, col)


    draw.save("output.2")

    return is_correctable(**locals())


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


def test_puncture(H):
    print("\ntest_puncture --------")
    n = H.shape[1]

    G = find_kernel(H)
    k = len(G)
    print("n = %d, k = %d" %(n, k))
    print("G =")
    print(shortstr(G))

    R = row_reduce(H)
    print("R =")
    print(shortstr(R))

    pivots = get_pivots(R)
    rows = [i for (i, j) in pivots]
    cols = [j for (i, j) in pivots]
    print("pivots:", pivots)
    A = cols[:k]
    remain = [j for j in range(n) if j not in A]
    S = R[:, remain]
    print("S =")
    print(shortstr(S))
    G1 = find_kernel(S)
    print("G1 =")
    G1 = row_reduce(G1)
    print(G1)
    S = row_reduce(S)
    print("S: rank = ", rank(S))
    print(shortstr(S))



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

    elif argv.hrand:
        #C = random_code(16, 8, 0, 3)
        C = random_code(8, 4, 0, 3)
        D = random_code(8, 4, 1, 3)

    elif argv.hvrand:
        #C = random_code(16, 8, 8, 3)
        C = random_code(8,  4, 4, 3)
        D = random_code(8,  4, 4, 3)

    elif argv.hvrandsmall:
        #C = random_code(16, 8, 8, 3)
        C = random_code(7,  1, 1, 2) # n, k, kt, d
        D = random_code(7,  1, 1, 2)

    elif argv.samerand:
        C = random_code(12, 6, 0, 4)
        D = C

    elif argv.smallrand:
        # make some vertical logops from rank degenerate parity check matrices
        C = random_code(8, 4, 0, 3)
        D = random_code(6, 3, 0, 2)

    elif argv.cookup:
        # [12,6,4] example that has no k-bipuncture
        C = parse("""
        .1..1111....
        111.11111111
        11.111...11.
        1..11...1.11
        .11...1...11
        ..11.11.111.
        """)
        D = C
        R = row_reduce(C)
        print(shortstr(R))

    elif argv.cookup2:
        # [12,6,4] example that has no k-bipuncture
        C = parse("""
        .11..1.11..1
        11...1111...
        1....1.11111
        ..1.1111..1.
        111....1.11.
        1111.11...11
        .1.1.1....1.
        1111111.1111
        ....1..111..
        .1..1.111.11
        11.11......1
        11..1111.1..
        """)
        D = C

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
        print("please specify a code")
        return

    print("C: shape=%s, rank=%d, dist=%d"%(C.shape, rank(C), classical_distance(C)))
    print("C.t: dist=%d"%(classical_distance(C.transpose()),))
    print(shortstr(C))
    print("D: shape=%s, rank=%d, dist=%d"%(D.shape, rank(D), classical_distance(D)))
    print("D.t: dist=%d"%(classical_distance(D.transpose()),))
    print(shortstr(D))
    Ct = C.transpose()
    Dt = D.transpose()

    if argv.dual:
        C, Ct = Ct, C
        D, Dt = Dt, D

    if argv.test_puncture:
        test_puncture(C)
        return # <--------- return

    if argv.test_indep:
        kw = hypergraph_product(C, Dt)
        test_indep(**kw)
        return # <--------- return

    if argv.test_code:
        kw = hypergraph_product(C, Dt)
        test_code(**kw)
        return # <--------- return

    if argv.test_overlap:

      #while 1:
        kw = hypergraph_product(C, Dt)
        success = test_overlap(**kw)

        print("success:", success)
        if argv.success:
            assert success

        #if success:
        #    break

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


