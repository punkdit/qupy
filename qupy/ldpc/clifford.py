#!/usr/bin/env python3

"""
A version of the qubit clifford group
over the ring D[i] where D is the dyadic rationals.

We use a modified Hadamard gate, see [1] page 31.

[1] Markus Heinrich PhD thesis at 
    https://kups.ub.uni-koeln.de/50465/

"""

print
from random import shuffle

import numpy
from numpy import dot, alltrue, zeros, array, identity, kron


from qupy.ldpc import asymplectic
from qupy.util import mulclose_fast, mulclose_hom, mulclose_names
from qupy.smap import SMap
from qupy.argv import argv


#int_scalar = numpy.int8 # dangerous...
int_scalar = numpy.int32 # less dangerous ?



def normalize(re, im, dyadic):
    while dyadic < 0:
        if not alltrue(re & 1 == 0) or not alltrue(im & 1 == 0):
            break
        re = re >> 1
        im = im >> 1
        dyadic += 1
    return re, im, dyadic


class Clifford(object):
    """
    Store Clifford matrix as real and imaginary
    integer matrices and a dyadic prefactor.
    """

    def __init__(self, re, im, dyadic=0, inv=None):
        assert dyadic <= 0, self
        if dyadic < 0:
            re, im, dyadic = normalize(re, im, dyadic)
        self.re = re
        self.im = im
        self.dyadic = dyadic # a factor of 2**dyadic
        self.inv = inv # track multiplicative _inverse

    def __str__(self):
        dyadic = self.dyadic
        s = "%s + \ni%s" % (self.re, self.im)
        if dyadic != 0:
            i = 2**dyadic
            s = "%s(%s)"%(i, s)
        return s
    __repr__ = __str__

    def __eq__(self, other):
        return (self.dyadic == other.dyadic and
            alltrue(self.re == other.re) and 
            alltrue(self.im == other.im))

    def __hash__(A):
        re, im, dyadic = A.re, A.im, A.dyadic
        key = re.tobytes(), im.tobytes(), dyadic
        return hash(key)

    def neg(A, inv=None):
        re, im, dyadic = A.re, A.im, A.dyadic
        op = Clifford(-re, -im, dyadic)
        op.inv = A.inv.neg(op) if inv is not None else inv # recurse
        return op
    __neg__ = neg

    def mul(A, B, inv=None):
        re = dot(A.re, B.re) - dot(A.im, B.im)
        im = dot(A.re, B.im) + dot(A.im, B.re)
        dyadic = A.dyadic + B.dyadic
        op = Clifford(re, im, dyadic)
        if inv is None and B.inv is not None and A.inv is not None:
            inv = B.inv.mul(A.inv, op) # recurse
        op.inv = inv
        return op
    __mul__ = mul

    def __matmul__(A, B):
        re = kron(A.re, B.re) - kron(A.im, B.im)
        im = kron(A.re, B.im) + kron(A.im, B.re)
        dyadic = A.dyadic + B.dyadic
        op = Clifford(re, im, dyadic)
        A, B = A.inv, B.inv
        re = kron(A.re, B.re) - kron(A.im, B.im)
        im = kron(A.re, B.im) + kron(A.im, B.re)
        dyadic = A.dyadic + B.dyadic
        op.inv = Clifford(re, im, dyadic, op)
        return op

    @classmethod
    def identity(cls, n):
        N = 2**n
        re = identity(N, dtype=int_scalar)
        im = zeros((N, N), dtype=int_scalar)
        I = cls(re, im)
        I.inv = I
        return I

    @classmethod
    def cz(cls, n, src=0, tgt=1):
        assert (n, src, tgt) == (2, 0, 1)
        N = 2**n
        re = identity(N, dtype=int_scalar)
        im = zeros((N, N), dtype=int_scalar)
        re[N-1, N-1] = -1
        op = cls(re, im)
        op.inv = op
        return op

    @classmethod
    def cx(cls, n, src=0, tgt=1):
        assert (n, src, tgt) == (2, 0, 1)
        N = 2**n
        re = zeros((N, N), dtype=int_scalar)
        im = zeros((N, N), dtype=int_scalar)
        re[:,:] = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
        op = cls(re, im)
        op.inv = op
        return op

    @classmethod
    def phase(cls, n):
        N = 2**n
        re = zeros((N, N), dtype=int_scalar)
        im = identity(N, dtype=int_scalar)
        op = cls(re, im)
        op.inv = cls(-re, -im, 0, op)
        return op

    @classmethod
    def zgate(cls):
        N = 2
        re = identity(N, dtype=int_scalar)
        re[1, 1] = -1
        im = zeros((N, N), dtype=int_scalar)
        op = cls(re, im)
        op.inv = op
        return op

    @classmethod
    def sgate(cls):
        N = 2
        re = zeros((N, N), dtype=int_scalar)
        im = zeros((N, N), dtype=int_scalar)
        re[0, 0] = 1
        im[1, 1] = 1
        op = cls(re, im)
        re = zeros((N, N), dtype=int_scalar)
        im = zeros((N, N), dtype=int_scalar)
        re[0, 0] = 1
        im[1, 1] = -1
        op.inv = cls(re, im, 0, op)
        return op

    @classmethod
    def xgate(cls):
        N = 2
        re = zeros((N, N), dtype=int_scalar)
        im = zeros((N, N), dtype=int_scalar)
        re[1, 0] = 1
        re[0, 1] = 1
        op = cls(re, im)
        op.inv = op
        return op

    @classmethod
    def ygate(cls):
        N = 2
        re = zeros((N, N), dtype=int_scalar)
        im = zeros((N, N), dtype=int_scalar)
        im[1, 0] = 1
        im[0, 1] = -1
        op = cls(re, im)
        op.inv = op
        return op

    @classmethod
    def hgate(cls):
        "This is the usual hadamard gate times a phase exp(i*pi/4)."
        N = 2
        re = array([[+1, +1], [+1, -1]], dtype=int_scalar)
        im = re.copy()
        op = cls(re, im, -1)
        re = re.copy()
        im = -re
        op.inv = cls(re, im, -1, op)
        return op




def main():

    I = Clifford.identity(1)
    iI = Clifford.phase(1)
    nI = iI * iI
    X = Clifford.xgate()
    Z = Clifford.zgate()
    Y = Clifford.ygate()
    S = Clifford.sgate()

    assert I != X != Z != S
    assert X*X == I
    assert Z*Z == I
    assert Y*Y == I
    assert X*Z != Z*X
    assert X*Z == nI*Z*X
    assert S*S == Z
    assert S*S*S*S == I
    assert S*X*S*X == iI

    H = Clifford.hgate()
    assert H*H == iI
    assert H*H*H*H == -I
    Hinv = H*H*H*H*H*H*H
    assert H*H.inv == I

    assert H*X*H.inv == Z
    assert H*Z*H.inv == X
    assert H*Y*H.inv == -Y

    gen = [H, S, X]
    G = mulclose_fast(gen)
    assert len(G) == 4*24

    for g in G:
        assert g * g.inv == I

    # -----------------------------------------------

    ASp = asymplectic.build()

    if 0:
        hom = mulclose_hom([H, S, X], [ASp.H, ASp.S, ASp.X])
    
        print("|Clifford| =", len(hom))
        kernel = []
        image = set()
        for g in hom:
            image.add(hom[g])
            if hom[g] == ASp.I:
                kernel.append(g)
        print("kernel:", len(kernel))
        print("image:", len(image))
    
        for g in hom:
          for h in hom:
            assert hom[g*h] == hom[g] * hom[h]

    # -----------------------------------------------

    II = Clifford.identity(2)
    iII = Clifford.phase(2)
    nII = iII * iII
    CX = Clifford.cx(2)
    CZ = Clifford.cz(2)

    IX = I @ X
    XI = X @ I
    IZ = I @ Z
    ZI = Z @ I
    IS = I @ S
    SI = S @ I
    IH = I @ H
    HI = H @ I
    XZ = XI*IZ
    ZX = IX*ZI

    assert CZ * CZ == II

    assert XI * XI == II
    assert Z@X == ZI*IX 

    assert CZ == IH * CX * IH.inv

    assert CZ * ZI == ZI * CZ
    assert CZ * IZ == IZ * CZ

    assert CZ * XI * CZ == XI*IZ
    assert CZ * IX * CZ == IX*ZI

    if argv.slow:
        gen = [IX, XI, SI, IS, HI, IH, CZ]
        G = mulclose_fast(gen)
        assert len(G) == 4*11520
        for g in G:
            assert g * g.inv == II

#    CZCX = CZ*CX
#    CXCZ = CX*CZ
#    phases = [II, iII, nII, nII*iII]
#    def table(op):
#        gen = [p*op for op in [II, XI, IX, ZI, IZ, XZ, ZX] for p in phases]
#        for src in gen:
#            tgt = op * src * op.inv
#            for idx, h in enumerate(gen):
#                if tgt == h:
#                    break
#            else:
#                assert 0
#    table(CZCX)
#    table(CXCZ)
#    return

    # -----------------------------------------------

    src = [IX, XI, SI, IS, HI, IH, CZ]
    ASp = asymplectic.build_stim()
    tgt = [ASp.IX, ASp.XI, ASp.SI, ASp.IS, ASp.HI, ASp.IH, ASp.CZ]
    G = mulclose_fast(tgt)
    hom = mulclose_hom(src, tgt)

    print("|Clifford| =", len(hom))
    kernel = []
    image = set()
    G = list(hom.keys())
    shuffle(G)
    for g in G:
        image.add(hom[g])
        if hom[g] == ASp.II:
            kernel.append(g)
    print("kernel:", len(kernel))
    print("image:", len(image))

    return

    for g in G:
      for h in G:
        assert hom[g*h] == hom[g] * hom[h]


def mulclose_hom_check(gen1, gen2, verbose=False, maxsize=None):
    "build a group hom from generators: gen1 -> gen2"
    hom = {}
    send = {}
    assert len(gen1) == len(gen2)
    for i in range(len(gen1)):
        hom[gen1[i]] = gen2[i]
    bdy = list(gen1)
    changed = True
    while bdy:
        #if verbose:
        #    print "mulclose:", len(hom)
        _bdy = []
        for A in gen1:
            for B in bdy:
                C1 = A*B
                if C1 not in hom:
                    hom[C1] = hom[A] * hom[B]
                    _bdy.append(C1)
                    if maxsize and len(els)>=maxsize:
                        return list(els)
                elif hom[C1] != hom[A] * hom[B]:
                    return None
                        
        bdy = _bdy
    return hom






if __name__ == "__main__":

    main()

    print("OK\n")


