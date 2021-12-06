#!/usr/bin/env python3

"""
Can we represent Clifford's as affine symplectic _transforms over Z/2 ?
(The qubit clifford group modulo phases.)
This does not seem to work, because of characteristic 2 glitches.

possible todo: implement phases as a group extension:
    phases >---> Clifford --->> AffineSymplectic

for qudit version see asymplectic_d.py

"""

from collections import namedtuple
from functools import reduce
from operator import mul
from random import shuffle
import string

import numpy
from numpy import concatenate as cat

from qupy.util import mulclose_fast
from qupy.tool import cross, choose
from qupy.argv import argv
from qupy.smap import SMap
from qupy.ldpc.solve import zeros2, shortstr, solve, dot2, array2, eq2, parse, pseudo_inverse, identity2
from qupy.ldpc.solve import enum2, row_reduce
from qupy.ldpc.css import CSSCode


def order(n):
    # size of Clifford group, modulo phases
    sz = 1
    for i in range(1, n+1):
        sz *= 2 * (4**1 - 1) * 4**i
    return sz

_cache = {}
def symplectic_form(n):
    F = _cache.get(n)
    if F is None:
        F = zeros2(2*n, 2*n)
        for i in range(n):
            F[n+i, i] = 1
            F[i, n+i] = 1
        _cache[n] = F
    F = F.copy()
    return F


class Clifford(object):
    """
    """

    def __init__(self, A):
        assert A.data.c_contiguous, "need to copy A first"
    
        m, n = A.shape
        assert m == n
        assert n%2 == 1
        self.shape = (n, n)
        self.n = n
        assert A[n-1, n-1] == 1, A
        assert numpy.abs(A[n-1]).sum() == 1, A

        self.A = A
        self.key = A.tobytes() # needs A to be c_contiguous 
        #assert self.check()

    def __str__(self):
        #s = str(self.A)
        #s = s.replace("0", ".")
        s = shortstr(self.A)
        return s

    def __mul__(self, other):
        assert isinstance(other, Clifford)
        assert other.shape == self.shape
        A = dot2(self.A, other.A)
        return Clifford(A)

    def __call__(self, v):
        assert len(v)+1 == self.n
        n = self.n
        A = self.A
        v = dot2(A[:n-1, :n-1], v)
        v = (v + A[:n-1, n-1]) % 2
        return v

    def __getitem__(self, key):
        return self.A[key]

    def __setitem__(self, key, value):
        assert 0, "do we really need this?"
        self.A[key] = value
        self.key = A.tobytes()

    def transpose(self):
        A = self.A.transpose().copy()
        return Clifford(A)

#    def inverse(self):
#        A = pseudo_inverse(self.A)
#        return Clifford(A)

    def inverse(self):
        A = self.A
        nn = self.n
        n = (nn-1)//2
        B = A[:2*n, :2*n] # symplectic 
        v = A[:2*n, 2*n]  # translation, shape (2*n,)
        F = symplectic_form(n)
        Fi = F # an involution
        Bi = dot2(Fi, dot2(B.transpose()), F)
        A1 = A.copy()
        A1[:2*n, :2*n] = Bi
        A1[:2*n, 2*n] = dot2(-Bi, v)
        return Clifford(A1)

    def __eq__(self, other):
        assert isinstance(other, Clifford)
        assert self.shape == other.shape
        #return eq2(self.A, other.A)
        return self.key == other.key

    def __ne__(self, other):
        assert isinstance(other, Clifford)
        assert self.shape == other.shape
        return self.key != other.key

    def __lt__(self, other):
        assert isinstance(other, Clifford)
        assert self.shape == other.shape
        return self.key < other.key

    def __hash__(self):
        # warning: i am mutable
        return hash(self.key)

    def check(self):
        # check symplectic condition
        A = self.A
        nn = self.n
        n = (nn-1)//2
        B = A[:2*n, :2*n]
        F = symplectic_form(n)
        lhs = dot2(dot2(B.transpose(), F), B)
        return numpy.alltrue(lhs == F)

    @classmethod
    def identity(cls, n):
        A = zeros2(2*n+1, 2*n+1)
        for i in range(2*n+1):
            A[i, i] = 1
        return Clifford(A)

    @classmethod
    def from_symplectic_and_translation(cls, A, v=None):
        n = len(A)
        assert n%2 == 0
        B = zeros2(n+1, n+1)
        B[n, n] = 1
        B[:n, :n] = A
        if v is not None:
            assert len(v) == n, len(v)
            B[:n, n] = v
        return Clifford(B)
    from_symplectic = from_symplectic_and_translation

    @classmethod
    def z(cls, n, idx):
        A = zeros2(2*n+1, 2*n+1)
        for i in range(2*n+1):
            A[i, i] = 1
        A[idx, 2*n] = 1
        return Clifford(A)

    @classmethod
    def x(cls, n, idx):
        A = zeros2(2*n+1, 2*n+1)
        for i in range(2*n+1):
            A[i, i] = 1
        A[n+idx, 2*n] = 1
        return Clifford(A)

    @classmethod
    def hadamard(cls, n, idx):
        A = zeros2(2*n+1, 2*n+1)
        for i in range(2*n):
            if i==idx:
                A[i, n+i] = 1
            elif i==n+idx:
                A[i, i-n] = 1
            else:
                A[i, i] = 1
        A[2*n, 2*n] = 1
        return Clifford(A)

    @classmethod
    def cnot(cls, n, src, tgt):
        assert src!=tgt
        A = zeros2(2*n+1, 2*n+1)
        for i in range(2*n+1):
            A[i, i] = 1
        A[2*n, 2*n] = 1
        A[src, tgt] = 1
        A[tgt+n, src+n] = 1
        return Clifford(A)

    @classmethod
    def cz(cls, n, src, tgt):
        CN = cls.cnot(n, src, tgt)
        H = cls.hadamard(n, tgt)
        CZ = H * CN * H
        return CZ

    @classmethod
    def s(cls, n, i):
        A = cls.identity(n).A
        assert 0<=i<n
        #A[i+n, i] = 1
        A[i, i+n] = 1
        A[i+n, 2*n] = 1 # ????
        return Clifford(A)
    
    @classmethod
    def swap(cls, n, idx, jdx):
        A = zeros2(2*n+1, 2*n+1)
        A[2*n, 2*n] = 1
        assert idx != jdx
        for i in range(n):
            if i==idx:
                A[i, jdx] = 1 # X sector
                A[i+n, jdx+n] = 1 # Z sector
            elif i==jdx:
                A[i, idx] = 1 # X sector
                A[i+n, idx+n] = 1 # Z sector
            else:
                A[i, i] = 1 # X sector
                A[i+n, i+n] = 1 # Z sector
        return Clifford(A)

    @classmethod
    def make_op(cls, v, method):
        assert method in [cls.x, cls.z, cls.s, cls.hadamard]
        n = len(v)
        g = Clifford.identity(n)
        for i in range(n):
            if v[i]:
                g *= method(n, i)
        return g

    def get_symplectic(self):
        n = self.n
        A = self.A
        B = A[:n-1, :n-1]
        return B

    def get_translation(self):
        n = self.n
        A = self.A
        v = A[:n-1, n-1]
        return v

    def is_translation(self):
        n = self.n
        A = self.A
        A = A[:n-1, :n-1]
        return numpy.alltrue(A == identity2(n-1))

    def get_order(self):
        I = Clifford.identity(self.n//2)
        op = self
        count = 1
        while op != I:
            op = self*op
            count += 1
        return count

    def find_encoding(A, code):
        Lx = [Clifford.make_op(l, Clifford.x) for l in code.Lx]
        Lz = [Clifford.make_op(l, Clifford.z) for l in code.Lz]
        Hx = [Clifford.make_op(l, Clifford.x) for l in code.Hx]
        Hz = [Clifford.make_op(l, Clifford.z) for l in code.Hz]
        assert len(Lx) == len(Lz)
        Ai = A.inverse()
        rows = []
        for l in Lz + Lx:
            B = A*l*Ai
            assert B.is_translation()
            v = B.get_translation()
            rows.append(v)
        B = numpy.array(rows)
        #B = B[:8, :]
        #print(B.shape)
        #print(shortstr(B))
        U = numpy.array([l.get_translation() for l in Lz+Lx+Hz+Hx])
        #print(U.shape)
        #print(shortstr(U))
        Ut = U.transpose()
        V = solve(Ut, B.transpose())
        if V is None:
            return None
        V = V[:2*code.k]
        #print(V.shape)
        #print(shortstr(V))

        # check that V is symplectic
        n = len(V) // 2
        F = symplectic_form(n)
        lhs = dot2(V.transpose(), dot2(F, V))
        assert eq2(lhs, F)

        #print(A.get_translation())
        u = solve(Ut, A.get_translation())
        u = u[:2*code.k]
        #print(shortstr(u))

        return Clifford.from_symplectic_and_translation(V, u)

    def is_transversal(A, code):
        Lz = [Clifford.make_op(l, Clifford.z) for l in code.Lz]
        Lx = [Clifford.make_op(l, Clifford.x) for l in code.Lx]
        Hz = [Clifford.make_op(l, Clifford.z) for l in code.Hz]
        Hx = [Clifford.make_op(l, Clifford.x) for l in code.Hx]
        hz = [op.get_translation() for op in Hz]
        hx = [op.get_translation() for op in Hx]
        h = array2(hz+hx)
        ht = h.transpose()
        Ai = A.inverse()
        tgt = []
        for u in Hz + Hx:
            v = A*u*Ai
            assert v.is_translation()
            v = v.get_translation()
            tgt.append(v)
            #print(u.get_translation())
            #print("-->")
            #print(v)
            #print()
        tgt = array2(tgt)
        src = array2([u.get_translation() for u in Hz+Hx])
        tgt, src = tgt.transpose(), src.transpose()
        if solve(tgt, src) is None:
            return False
        if solve(src, tgt) is None:
            return False
        return True


class NS(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)



def build():

    # --------------------------------------------
    # Clifford group order is 24

    n = 1
    I = Clifford.identity(n)

    X = Clifford.x(n, 0)
    Z = Clifford.z(n, 0)
    S = Clifford.s(n, 0)
    Si = S.inverse()
    H = Clifford.hadamard(n, 0)
    Y = X*Z

    #make_relators(I, [X, Z, S, H, S*H]) # works
    #return

    # --------------------------------------------
    # Clifford group order is 11520

    n = 2
    II = Clifford.identity(n)
    XI = Clifford.x(n, 0)
    IX = Clifford.x(n, 1)
    ZI = Clifford.z(n, 0)
    IZ = Clifford.z(n, 1)
    SI = Clifford.s(n, 0)
    IS = Clifford.s(n, 1)
    HI = Clifford.hadamard(n, 0)
    IH = Clifford.hadamard(n, 1)
    CX = Clifford.cnot(n, 0, 1)
    CZ = Clifford.cz(n, 0, 1)

    #assert CZ*SI == SI*CZ # FAIL
    #assert CZ*IS == IS*CZ # FAIL
    assert CZ*CZ == II

##    front = [XI, ZI, SI, HI, SI*HI, HI*SI*HI, CZ, SI*CZ, IS*CZ]
##    front = [XI, ZI, SI, HI, SI*HI, HI*SI*HI, CZ, XI*CZ, IX*CZ]
#    front = [XI, ZI, SI, HI, SI*HI, HI*SI*HI, CZ, XI*CZ, IX*CZ, CZ*XI*CZ, CZ*IX*CZ]
#    back = [II, IX, IZ, IS, IH, IS*IH, IH*IS*IH]
#    #gen = [g*h for g in front for h in back]
#    gen = front + back 
#    make_relators(II, gen, depth=6)

    del n

    return NS(**locals())


def contains(src, tgt):
    n, m = len(src), len(tgt)
    if n > m:
        return False
    for i in range(m-n+1):
        if src == tgt[i:i+n]:
            return True
    return False


def make_relators(I, gen, names=None, depth=5):
    # make some gap code for generators and relators
    n = len(gen)
    rels = set()
    def addword(word):
        i = min(word)
        idx = word.index(i)
        n = len(word)
        word = tuple(word[(i+idx)%n] for i in range(n))
        rels.add(word)

    for width in range(2, depth):
        itemss = [tuple(range(n))]*width
        for word in cross(itemss):
            gword = [gen[i] for i in word]
            g = reduce(mul, gword)
            if g == I:
                for h in rels:
                    if contains(h, word+word):
                        break
                else:
                    addword(word)
    print("rels:", len(rels))
    if names is None:
        names = string.ascii_letters[:n]
    if len(names) < n:
        names = ["%s%s"%(a,b) for a in string.ascii_letters for b in string.ascii_letters]
    assert len(names) == len(set(names)) == n, repr(names)
    print()
    print("F := FreeGroup(%s);;" % (",".join('"%s"'%l for l in names)))
    for idx, name in enumerate(names):
        print("%s := F.%s;;" % (name, idx+1))
    items = []
    rels = list(rels)
    rels.sort()
    for rel in rels:
        rel = [names[idx] for idx in rel]
        rel = "*".join(rel)
        items.append(rel)
    print("G := F / [%s];;" % (','.join(items)))
    print("Order(G);")


def mulclose_names(gen, names, verbose=False, maxsize=None):
    ops = list(gen)
    words = dict((g, names[i]) for (i, g) in enumerate(gen))
    bdy = gen
    while bdy:
        _bdy = []
        for g in bdy:
            for h in gen:
                k = g*h
                try:
                    idx = ops.index(k)
                    if len(words[g]+words[h]) < len(words[ops[idx]]):
                        words[ops[idx]] = words[g]+words[h]
                except ValueError:
                    words[k] = words[g]+words[h]
                    ops.append(k)
                    _bdy.append(k)
        bdy = _bdy
        if verbose:
            print("mulclose:", len(ops))
        if maxsize and len(ops) >= maxsize:
            break
    return ops, words


class Stim(object):
    def __init__(self, op):
        #self.op = op
        #self.key = str(op)
        self.n = len(op)
        self.key = repr(op)
        self._hash = hash(self.key)
    @property
    def op(self):
        import stim
        op = eval(self.key)
        return op

#class Stim(object):
#    def __init__(self, op):
#        r = repr(op)
#        print(r)
#        flds = r.split('"')
#        assert len(flds) == 5, flds
#        x = flds[1]
#        z = flds[3]
#        self.key = (x, z)
#        self._hash = hash(self.key)
#    @property
#    def op(self):
#        from stim import Tableau, PauliString
#        x, z = self.key
#        op = Tableau.from_conjugated_generators( 
#            xs=[PauliString(x)], zs=[PauliString(z)])
#        return op

    def __eq__(self, other):
        return self.op == other.op

    def __hash__(self):
        #return hash(str(self.op))
        return self._hash

    def __mul__(self, other):
        return Stim(self.op * other.op)

    def inverse(self):
        return Stim(self.op**-1)

    def __call__(self, phase, hz, hx):
        # Here we represent pauli ops as phase*Z*X
        # so that Y = i*Z*X
        from stim import PauliString
        n = self.n
        src = PauliString("I"*n)
        #print("__call__", phase, hz, hx)
        if hz is not None:
            assert n == len(hz)
            src = src*PauliString(''.join(["IZ"[i] for i in hz]))
        #print("\t", src)
        if hx is not None:
            assert n == len(hx)
            src = src*PauliString(''.join(["IX"[i] for i in hx]))
        #print("\t", src)
        tgt = self.op(src)
        #print("\t", phase, "*", src, "--->", tgt)
        phase *= tgt.sign
        zop, xop = [0]*n, [0]*n
        for i in range(n):
            j = tgt[i] # 0=I, 1=X, 2=Y, 3=Z
            if j==0:
                pass
            elif j==1:
                xop[i] = 1
            elif j==2:
                # Y = -1j*Z*X
                xop[i] = 1
                zop[i] = 1
                phase *= -1j
            elif j==3:
                zop[i] = 1
            else:
                assert 0, "j=%d"%j
        return phase, zop, xop

    @classmethod
    def identity(cls, n):
        from stim import Tableau
        op = Tableau(n)
        return Stim(op)

    @classmethod
    def mkop(cls, n, idx, name):
        from stim import Tableau
        op = Tableau(n)
        gate = Tableau.from_named_gate(name)
        op.append(gate, [idx])
        return Stim(op)

    @classmethod
    def x(cls, n, idx):
        return cls.mkop(n, idx, "X")
    x_gate = x

    @classmethod
    def z(cls, n, idx):
        return cls.mkop(n, idx, "Z")
    z_gate = z

    @classmethod
    def s(cls, n, idx):
        return cls.mkop(n, idx, "S")
    s_gate = s

    @classmethod
    def hadamard(cls, n, idx):
        return cls.mkop(n, idx, "H")
    h_gate = hadamard

    @classmethod
    def cnot(cls, n, src, tgt):
        assert 0<=src<n
        assert 0<=tgt<n
        from stim import Tableau
        gate = Tableau.from_named_gate("CNOT")
        op = Tableau(n)
        op.append(gate, [src, tgt])
        return cls(op)
    cx_gate = cnot

    @classmethod
    def cz(cls, n, src, tgt):
        assert 0<=src<n
        assert 0<=tgt<n
        from stim import Tableau
        gate = Tableau.from_named_gate("CZ")
        op = Tableau(n)
        op.append(gate, [src, tgt])
        return cls(op)
    cz_gate = cz

    @classmethod
    def swap(cls, n, src, tgt):
        assert 0<=src<n
        assert 0<=tgt<n
        from stim import Tableau
        gate = Tableau.from_named_gate("SWAP")
        op = Tableau(n)
        op.append(gate, [src, tgt])
        return cls(op)
    swap_gate = swap



def build_stim():
    from stim import PauliString

    X = PauliString("X")
    Y = PauliString("Y")
    Z = PauliString("Z")
    assert str(Z*X) == "+iY"
    assert str(X*Z) == "-iY"
    assert str(X*Y*Z) == "+i_"

    n = 1
    S = Stim.s_gate(n, 0)
    Si = S.inverse()

    op = (1., [0], [1]) # == +X
    op = S(*op)
    assert op == (-1j, [1], [1]) # == Y
    op = S(*op)
    assert op == (-1, [0], [1]) # == -X
    op = S(*op)
    assert op == (+1j, [1], [1]) # == Y
    op = S(*op)
    assert op == (+1, [0], [1]) # == +X

    op = (1., [0], [1])
    assert Si(*op) == (1j, [1], [1]) # == -Y

    CZ = Stim.cz_gate(2, 0, 1)
    print(CZ(1., [0,0], [1,1]))

    # --------------------------------------------
    # Clifford group order is 11520

    n = 2
    II = Stim.identity(n)
    XI = Stim.x(n, 0)
    IX = Stim.x(n, 1)
    ZI = Stim.z(n, 0)
    IZ = Stim.z(n, 1)
    SI = Stim.s(n, 0)
    IS = Stim.s(n, 1)
    HI = Stim.hadamard(n, 0)
    IH = Stim.hadamard(n, 1)
    CX = Stim.cnot(n, 0, 1)
    CZ = Stim.cz(n, 0, 1)
    XX = XI*IX
    ZZ = ZI*IZ
    XZ = XI*IZ
    ZX = IX*ZI
    YI = XI*ZI
    IY = IX*IZ
    YY = YI*IY
    XY = XI*IY
    YX = IX*YI
    ZY = ZI*IY
    YZ = IZ*YI

    assert XI*ZI == ZI*XI
    assert SI*SI == ZI
    assert SI*ZI == ZI*SI
    assert SI*XI != XI*SI
    assert SI*SI*SI*SI == II
    assert CX * CX == II
    assert CZ * CZ == II
    assert CX * IX == IX * CX
    assert CX * ZI == ZI * CX

    assert CX * IZ * CX == ZZ
    assert CX * ZZ * CX == IZ
    assert CX * XI * CX == XX
    assert CZ * XX * CZ == YY
    assert CX * XZ * CX == YY

    assert CZ == IH * CX * IH
    assert CZ * ZI == ZI * CZ
    assert CZ * IZ == IZ * CZ
    assert CZ * XI * CZ == XZ
    assert CZ * IX * CZ == ZX
    assert CZ * ZX * CZ == IX

    assert CZ*SI == SI*CZ
    assert CZ*IS == IS*CZ
    assert CZ*CZ == II

#    assert CX * CZ == CZ * CX

#    print("CZ:")
#    print(CZ.op)
#    print("CX:")
#    print(CX.op)
#    print("CX*CZ:")
#    print((CX*CZ).op)
#    print("CZ*CX:")
#    print((CZ*CX).op)

    # slow...
    #G = mulclose_fast([SI, IS, CX, HI, IH ])
    #assert len(G) == 11520, len(G)

    # --------------------------------------------

    n = 3
    III = Stim.identity(n)
    XII = Stim.x_gate(n, 0)
    IXI = Stim.x_gate(n, 1)
    IZI = Stim.z_gate(n, 1)
    ICZ = Stim.cz_gate(n, 1, 2)
    CZ_01 = Stim.cz_gate(n, 0, 1)

    SW_02 = Stim.swap_gate(n, 0, 2)
    CX_02 = Stim.cx_gate(n, 0, 2)
    CX_20 = Stim.cx_gate(n, 2, 0)

    assert ICZ == Stim.cz_gate(n, 2, 1)
    assert ICZ * ICZ == III
    assert XII * ICZ == ICZ * XII
    assert IXI * ICZ != ICZ * IXI
    assert IZI * ICZ == ICZ * IZI

    assert SW_02 == CX_02*CX_20*CX_02
    assert SW_02*SW_02 == III
    assert SW_02*CZ_01*SW_02 == ICZ
    assert SW_02.inverse() == SW_02

    g = CX_02 * CZ_01
    assert g * g.inverse() == III

    # ------------------------------------------

    n = 4
    cz01 = Stim.cz_gate(n, 0, 1)
    cz02 = Stim.cz_gate(n, 0, 2)
    cz03 = Stim.cz_gate(n, 0, 3)
    cz12 = Stim.cz_gate(n, 1, 2)
    cz13 = Stim.cz_gate(n, 1, 3)
    cz23 = Stim.cz_gate(n, 2, 3)
    h0 = Stim.h_gate(n, 0)
    h1 = Stim.h_gate(n, 1)
    h2 = Stim.h_gate(n, 2)
    h3 = Stim.h_gate(n, 3)
    s0 = Stim.s_gate(n, 0)
    s1 = Stim.s_gate(n, 1)
    s2 = Stim.s_gate(n, 2)
    s3 = Stim.s_gate(n, 3)

    gens = [
        h0*h1*h2*h3,
        cz01*cz23,
        cz02*cz13,
        cz03*cz12,         # = 768
        #Stim.z_gate(n, 0),
        #Stim.z_gate(n, 1),
        #Stim.z_gate(n, 2),
        #Stim.z_gate(n, 3), # = 196608
        #cz01*s2*s3,
        #cz02*s1*s3,
        #cz03*s1*s2,
    ]
    #G = mulclose_fast(gens, verbose=True)
    #print("\n|G| =", len(G))

    return NS(**locals())



def test():

    n = 3
    F = symplectic_form(n)
    I = identity2(2*n)
    assert numpy.alltrue(I == dot2(F, F))

    # --------------------------------------------
    # Clifford group order is 24

    n = 1
    I = Clifford.identity(n)

    X = Clifford.x(n, 0)
    Z = Clifford.z(n, 0)
    S = Clifford.s(n, 0)
    Si = S.inverse()
    H = Clifford.hadamard(n, 0)
    Y = X*Z

    print("X =")
    print(X)
    print("Z =")
    print(Z)
    print("S =")
    print(S)
    print("Si =")
    print(Si)
    print()

    assert X != I
    assert Z != I
    assert X*X == I
    assert Z*Z == I
    assert Z*X == X*Z # looses the phase 

    assert Si*S == S*Si == I
    assert Si*Si == Z
    assert S*Z == Z*S
    assert S*X == Y*S
    assert S*Y*Si == X
    assert S*S == Z
    assert S*X != X*S
    assert S*S*S*S == I
    assert S*S*S == Si

    assert H*H == I
    assert H*X*H == Z
    assert H*Z*H == X

    assert S*H*S*H*S*H == I

    G = mulclose_fast([S, H])
    assert len(G) == 24

#    # --------------------------------------------
#    # Look for group central extension
#
#    G = list(G)
#    G.sort()
#    P = [g for g in G if g.is_translation()] # Pauli group
#    N = len(G)
#    lookup = dict((g, idx) for (idx, g) in enumerate(G))
#    coord = lambda g, h : lookup[h] + N*lookup[g] 
#    #phi = numpy.zeros((N, N), dtype=int)
#
#    phi = {}
#    for g in P:
#      for h in P:
#        phi[g, h] = 0
#
#    pairs = [(g, h) for g in G for h in G]
#    triples = [(g, h, k) for g in G for h in G for k in G]
#    done = False
#    while not done:
#      done = True
#      print(len([phi.get(k) for k in pairs if phi.get(k) is None]))
#      for (g, h, k) in triples:
#        vals = [
#            phi.get((g, h)),
#            phi.get((h, k)),
#            phi.get((g, h*k)),
#            phi.get((g*h, k))]
#        if vals.count(None) == 1:
#            phi[g, h] = 0
#            phi[h, k] = 0
#            phi[g, h*k] = 0
#            phi[g*h, k] = 0
#            done = False
#    print(len([phi.get(k) for k in pairs if phi.get(k) is None]))
#    print(len(phi))
#    print(len(pairs))
#
#    return

    # --------------------------------------------
    # Clifford group order is 11520

    n = 2
    II = Clifford.identity(n)

    XI = Clifford.x(n, 0)
    IX = Clifford.x(n, 1)
    ZI = Clifford.z(n, 0)
    IZ = Clifford.z(n, 1)
    SI = Clifford.s(n, 0)
    IS = Clifford.s(n, 1)
    SS = SI * IS
    HI = Clifford.hadamard(n, 0)
    IH = Clifford.hadamard(n, 1)
    XX = XI*IX
    ZZ = ZI*IZ
    XZ = XI*IZ
    ZX = IX*ZI
    YI = XI*ZI
    IY = IX*IZ
    YY = YI*IY
    XY = XI*IY
    YX = IX*YI
    ZY = ZI*IY
    YZ = IZ*YI

    CX = Clifford.cnot(n, 0, 1)
    CX1 = Clifford.cnot(n, 1, 0)
    CZ = Clifford.cz(n, 0, 1)
    CZ1 = Clifford.cz(n, 1, 0)

    print("CZ =")
    print(CZ)

    print("CX =")
    print(CX)

    assert SI*SI == ZI

    assert SI*ZI == ZI*SI
    assert SI*XI != XI*SI
    assert SI*SI*SI*SI == II

    assert CX * CX == II
    assert CZ * CZ == II
    assert CZ1 == CZ

    assert CX * IX == IX * CX
    assert CX * XI * CX == XX

    assert CX * ZI == ZI * CX
    assert CX * IZ * CX == ZZ

    SWAP = Clifford.swap(n, 0, 1)
    assert SWAP * ZI == IZ * SWAP
    assert SWAP * XI == IX * SWAP

    assert CX * CX1 * CX == SWAP

    assert CZ == IH * CX * IH

    assert CZ * ZI == ZI * CZ
    assert CZ * IZ == IZ * CZ

    assert CZ * XI * CZ == XI*IZ
    assert CZ * IX * CZ == IX*ZI

    assert CX * IZ * CX == ZZ
    assert CX * ZZ * CX == IZ
    assert CX * XI * CX == XX
    assert CZ * XX * CZ == YY
    assert CX * XZ * CX == YY

    assert CX * CZ == CZ * CX
    print("CX * CZ =")
    print(CX * CZ)

    #print(CX*CX1)
    #print()
    #print(SWAP)

    G = mulclose_fast([SI, IS, CX, HI, IH ])
    assert len(G) == 11520

    for g in G:
        assert g.check()
        h = g.inverse()
        assert h.check()
        assert g*h == II

    # --------------------------------------------

    n = 5
    I = Clifford.identity(n)
    CZ = Clifford.cz(n, 0, 1)
    SWAP = Clifford.swap(n, 0, 1)
    assert CZ*CZ == I
    assert SWAP*CZ == CZ*SWAP



if __name__ == "__main__":

    name = argv.next()

    if argv.profile:
        import cProfile as profile
        profile.run("%s()"%name)

    elif name:

        fn = eval(name)
        fn()

    else:

        test()

    print("OK")



