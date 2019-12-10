#!/usr/bin/env python3

from collections import namedtuple
from functools import reduce
from operator import mul

import numpy
from numpy import concatenate as cat

from qupy.util import mulclose_fast
from qupy.argv import argv
from qupy.smap import SMap
from qupy.ldpc.solve import zeros2, shortstr, solve, dot2, array2, eq2, parse, pseudo_inverse
from qupy.ldpc.css import CSSCode


def mulclose_find(gen, names, target, verbose=False, maxsize=None):
    ops = set(gen)
    lookup = dict((g, (names[i],)) for (i, g) in enumerate(gen))
    bdy = gen
    dist = 1
    found = False
    while bdy:
        _bdy = set()
        for g in bdy:
            for h in gen:
                k = g*h
                if k in ops:
                    if len(lookup[g]+lookup[h]) < len(lookup[k]):
                        lookup[k] = lookup[g]+lookup[h]
                        assert 0
                    #continue

                else:
                    word = lookup[g]+lookup[h]
                    if len(word) > dist:
                        dist = len(word)
                        if verbose:
                            print("dist:", dist)
                        if found:
                            return
                    lookup[k] = word
                    ops.add(k)
                    _bdy.add(k)

                if k==target:
                    yield lookup[g]+lookup[h]
                    found = True
        bdy = _bdy
        #if verbose:
        #    print("mulclose:", len(ops))
        if maxsize and len(ops) >= maxsize:
            break


def mulclose_names(gen, names, verbose=False, maxsize=None):
    ops = set(gen)
    lookup = dict((g, (names[i],)) for (i, g) in enumerate(gen))
    bdy = gen
    while bdy:
        _bdy = set()
        for g in bdy:
            for h in gen:
                k = g*h
                if k in ops:
                    if len(lookup[g]+lookup[h]) < len(lookup[k]):
                        lookup[k] = lookup[g]+lookup[h]
                        assert 0
                else:
                    lookup[k] = lookup[g]+lookup[h]
                    ops.add(k)
                    _bdy.add(k)
        bdy = _bdy
        if verbose:
            print("mulclose:", len(ops))
        if maxsize and len(ops) >= maxsize:
            break
    return ops, lookup




class Clifford(object):
    def __init__(self, A):
        self.A = A
        m, n = A.shape
        self.shape = A.shape
        assert n%2 == 0
        self.n = n//2 # qubits
        self.key = A.tostring()

    def __str__(self):
        return str(self.A)

    def __mul__(self, other):
        assert isinstance(other, Clifford)
        A = dot2(self.A, other.A)
        return Clifford(A)

    def __call__(self, other):
        assert isinstance(other, CSSCode)
        assert other.n == self.n
        Lx, Lz, Hx, Tz, Hz, Tx, Gx, Gz = (
            other.Lx, other.Lz, other.Hx,
            other.Tz, other.Hz, other.Tx,
            other.Gx, other.Gz)
        assert Gx is None
        assert Gz is None
        A = self.A.transpose()
        LxLz = dot2(cat((Lx, Lz), axis=1), A)
        HxTz = dot2(cat((Hx, Tz), axis=1), A)
        TxHz = dot2(cat((Tx, Hz), axis=1), A)
        n = self.n
        Lx, Lz = LxLz[:, :n], LxLz[:, n:]
        Hx, Tz = HxTz[:, :n], HxTz[:, n:]
        Tx, Hz = TxHz[:, :n], TxHz[:, n:]
        code = CSSCode(Lx=Lx, Lz=Lz, Hx=Hx, Tz=Tz, Hz=Hz, Tx=Tx)
        return code

    def transpose(self):
        A = self.A.transpose().copy()
        return Clifford(A)

    def inverse(self):
        A = pseudo_inverse(self.A)
        return Clifford(A)

#    def __call__(self, other):
#        assert isinstance(other, CSSCode)
#        assert other.n == self.n
#        A = self.A.transpose()
#        B = other.to_symplectic()
#        C = dot2(B, A)
#        code = CSSCode.from_symplectic(C, other.n, other.k, other.mx, other.mz)
#        return code

    def __eq__(self, other):
        assert isinstance(other, Clifford)
        assert self.shape == other.shape
        #return eq2(self.A, other.A)
        return self.key == other.key

    def __ne__(self, other):
        return self.key != other.key

    def __hash__(self):
        return hash(self.key)

    @classmethod
    def identity(cls, n):
        A = zeros2(2*n, 2*n)
        for i in range(2*n):
            A[i, i] = 1
        return Clifford(A)

    @classmethod
    def hadamard(cls, n, idx):
        A = zeros2(2*n, 2*n)
        for i in range(2*n):
            if i==idx:
                A[i, n+i] = 1
            elif i==n+idx:
                A[i, i-n] = 1
            else:
                A[i, i] = 1
        return Clifford(A)

    @classmethod
    def cnot(cls, n, src, tgt):
        A = cls.identity(n).A
        assert src!=tgt
        A[tgt, src] = 1
        A[src+n, tgt+n] = 1
        return Clifford(A)



class Surface(object):

    def __init__(self):
        self.keys = []
        self.keymap = {}

    @property
    def n(self):
        return len(self.keys)

    def _add(self, key):
        keys = self.keys
        keymap = self.keymap
        if key in keymap:
            return
        keymap[key] = len(keys)
        keys.append(key)

    def add(self, top_left, bot_right):

        i0, j0 = top_left
        i1, j1 = bot_right

        # rough at top and bot
        # smooth at left and right
        for i in range(i0, i1):
          for j in range(j0, j1):
            ks = (0, 1)
            if i==i0:
                ks = (1,)
            if j==j1-1:
                ks = (1,)
            for k in ks:
                key = (i, j, k)
                self._add(key)

    def get_coord(self, i, j, k=None):
        if k is not None:
            assert k==0 or k==1
            row = 2*i + k
            col = 2*j + (1-k)
        else:
            row = 2*i
            col = 2*j
        return (row, col)

    def mk_smap(self):
        smap = SMap()
        get_coord = self.get_coord
        for (i, j, k) in self.keys:
            smap[get_coord(i, j)] = "o"
            if k==0:
                smap[get_coord(i, j, k)] = "-"
            elif k==1:
                smap[get_coord(i, j, k)] = "|"
        return smap

    def __str__(self):
        smap = self.mk_smap()
        return str(smap)

    def str_span(self, span, c="*"):
        smap = self.mk_smap()
        for (i, j, k) in span:
            smap[self.get_coord(i, j, k)] = c
        return str(smap)

    def str_op(self, op, c="*"):
        smap = self.mk_smap()
        for idx, key in enumerate(self.keys):
            if op[idx]==0:
                continue
            (i, j, k) = key
            smap[self.get_coord(i, j, k)] = c
        return str(smap)

    def get_ops(self):
        keys = self.keys
        keymap = self.keymap
        i0 = min(i for (i,j,k) in keys)
        i1 = max(i for (i,j,k) in keys)
        j0 = min(j for (i,j,k) in keys)
        j1 = max(j for (i,j,k) in keys)

        z_ops = []
        deltas = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 0, 0)]
        for i in range(i0, i1+1):
          for j in range(j0, j1+1):
            span = []
            for di, dj, k in deltas:
                key = (i+di, j+dj, k)
                if key not in keymap:
                    continue
                span.append(key)
            if len(span)>=3:
                z_ops.append(span)

        x_ops = []
        deltas = [(0, 0, 0), (0, 0, 1), (-1, 0, 1), (0, -1, 0)]
        for i in range(i0, i1+1):
          for j in range(j0, j1+1):
            span = []
            for di, dj, k in deltas:
                key = (i+di, j+dj, k)
                if key not in keymap:
                    continue
                span.append(key)
            if len(span)>=3:
                x_ops.append(span)

        return x_ops, z_ops

    def get_code(self):
        keys = self.keys
        keymap = self.keymap
        x_ops, z_ops = self.get_ops()
        n = len(keys)
        mx = len(x_ops)
        mz = len(z_ops)
        Hx = zeros2(mx, n)
        Hz = zeros2(mz, n)
        for idx, span in enumerate(x_ops):
            for key in span:
                Hx[idx, keymap[key]] = 1
        for idx, span in enumerate(z_ops):
            for key in span:
                Hz[idx, keymap[key]] = 1
        code = CSSCode(Hx=Hx, Hz=Hz)
        return code

    def dump_idxs(self):
        n = self.n
        keys = self.keys
        sep = '.'*2*self.n
        for idx in range(n):
            smap = self.mk_smap()
            (i, j, k) = keys[idx]
            smap[self.get_coord(i, j, k)] = str(idx)
            print(smap)
            print(sep)

    def dump_code(self, code):
        assert code.n == self.n
        str_op = self.str_op
        w = 2*self.n
        print("Lx:")
        for op in code.Lx:
            print(str_op(op, "X"))
            print("."*w)
        print("Lz:")
        for op in code.Lz:
            print(str_op(op, "Z"))
            print("."*w)
        print("Hx:")
        for op in code.Hx:
            print(str_op(op, "X"))
            print("."*w)
        print("Hz:")
        for op in code.Hz:
            print(str_op(op, "Z"))
            print("."*w)


def get_gen(n, pairs=None):
    #gen = [Clifford.hadamard(n, i) for i in range(n)]
    #names = ["H_%d"%i for i in range(n)]
    gen = []
    names = []
    if pairs is None:
        pairs = []
        for i in range(n):
          for j in range(n):
            if i!=j:
                pairs.append((i, j))
    for (i, j) in pairs:
        assert i!=j
        gen.append(Clifford.cnot(n, i, j))
        names.append("CN(%d,%d)"%(i,j))

    return gen, names


def test_symplectic():

    n = 4
    I = Clifford.identity(n)
    H = Clifford.hadamard(n, 0)
    assert H*H == I

    CN = Clifford.cnot(n, 0, 1)
    assert CN*CN == I

    n = 3
    trivial = CSSCode(
        Lx=parse("1.."), Lz=parse("1.."), Hx=zeros2(0, n), Hz=parse(".1. ..1"))

    assert trivial.row_equal(CSSCode.get_trivial(3, 0))

    repitition = CSSCode(
        Lx=parse("111"), Lz=parse("1.."), Hx=zeros2(0, n), Hz=parse("11. .11"))

    assert not trivial.row_equal(repitition)

    CN_01 = Clifford.cnot(n, 0, 1)
    CN_12 = Clifford.cnot(n, 1, 2)
    CN_21 = Clifford.cnot(n, 2, 1)
    CN_10 = Clifford.cnot(n, 1, 0)
    encode = CN_12 * CN_01

    code = CN_01 ( trivial )
    assert not code.row_equal(repitition)
    code = CN_12 ( code )
    assert code.row_equal(repitition)

#    assert (CN_21 * CN_10)(trivial).row_equal(repitition)
#    assert (CN_10 * CN_21)(trivial).row_equal(repitition)

#    src = Clifford(trivial.to_symplectic())
#    src_inv = src.inverse()
#    tgt = Clifford(repitition.to_symplectic())
#    A = (src_inv * tgt).transpose()
#    #print(A)

    A = get_encoder(trivial, repitition)

    gen, names = get_gen(3)
    for word in mulclose_find(gen, names, A):
        #print("word:")
        #print(word)
    
        items = [gen[names.index(op)] for op in word]
        op = reduce(mul, items)
    
        #print(op)
        #assert op*(src) == (tgt)
    
        #print(op(trivial).longstr())
        assert op(trivial).row_equal(repitition)
    

def get_encoder(source, target):
    assert isinstance(source, CSSCode)
    assert isinstance(target, CSSCode)
    src = Clifford(source.to_symplectic())
    src_inv = src.inverse()
    tgt = Clifford(target.to_symplectic())
    A = (src_inv * tgt).transpose()
    return A


def test_surface_1():

    surf = Surface()
    surf.add((0, 0), (2, 2))
    print(surf)

#    x_ops, z_ops = surf.get_ops()
#    for span in x_ops:
#        print()
#        print(surf.str_span(span, "X"))
#    for span in z_ops:
#        print()
#        print(surf.str_span(span, "Z"))

    code = surf.get_code()
    n = code.n
    print(code)
    print(code.longstr())

    idx = surf.keymap[(1, 0, 0)]
    trivial = CSSCode.get_trivial(surf.n, idx)

    if 0:
        op = Clifford.identity(n)
        for i in [0, 1, 3, 4]:
            op = op*Clifford.hadamard(n, i)
        trivial = op(trivial)

    A = get_encoder(trivial, code)

    pairs = None
    #pairs = [(0, 2), (1, 2), (3, 2), (4, 2)]
    pairs = [(0, 2), (1, 2), (3, 2), (4, 2),
        (2, 0), (2, 1), (2, 3), (2, 4)]
    gen, names = get_gen(surf.n, pairs)

    #G = mulclose_fast(gen)
    for result in mulclose_find(gen, names, A):
        print("*".join(result))


def test_surface():

    source = Surface()
    source.add((1, 1), (3, 3))
    print(source)
    print()

    target = Surface()
    target.add((0, 0), (4, 4))
    print(target)

    



if __name__ == "__main__":

    test_symplectic()
    test_surface()




