#!/usr/bin/env python3

from collections import namedtuple
from functools import reduce
from operator import mul

import numpy
from numpy import concatenate as cat

from qupy.util import mulclose_fast
from qupy.argv import argv
from qupy.smap import SMap
from qupy.ldpc.solve import zeros2, shortstr, solve, dot2, array2, eq2, parse, pseudo_inverse, identity2
from qupy.ldpc.solve import enum2
from qupy.ldpc.css import CSSCode
from qupy.ldpc.decoder import StarDynamicDistance


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




class Symplectic(object):
    def __init__(self, A):
        self.A = A
        m, n = A.shape
        self.shape = A.shape
        assert n%2 == 0
        self.n = n//2 # qubits
        self.key = A.tostring()

    def __str__(self):
        #s = str(self.A)
        #s = s.replace("0", ".")
        s = shortstr(self.A)
        return s

    def __mul__(self, other):
        assert isinstance(other, Symplectic)
        A = dot2(self.A, other.A)
        return Symplectic(A)

    def __getitem__(self, key):
        return self.A[key]

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
        return Symplectic(A)

    def inverse(self):
        A = pseudo_inverse(self.A)
        return Symplectic(A)

#    def __call__(self, other):
#        assert isinstance(other, CSSCode)
#        assert other.n == self.n
#        A = self.A.transpose()
#        B = other.to_symplectic()
#        C = dot2(B, A)
#        code = CSSCode.from_symplectic(C, other.n, other.k, other.mx, other.mz)
#        return code

    def __eq__(self, other):
        assert isinstance(other, Symplectic)
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
        return Symplectic(A)

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
        return Symplectic(A)

    @classmethod
    def cnot(cls, n, src, tgt):
        A = cls.identity(n).A
        assert src!=tgt
        A[tgt, src] = 1
        A[src+n, tgt+n] = 1
        return Symplectic(A)
    
    @classmethod
    def swap(cls, n, idx, jdx):
        A = zeros2(2*n, 2*n)
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
        return Symplectic(A)

    @classmethod
    def bilinear_form(cls, n):
        A = zeros2(2*n, 2*n)
        I = identity2(n)
        A[:n, n:] = I
        A[n:, :n] = I
        A = Symplectic(A)
        return A

    @classmethod
    def transvect(cls, x):
        assert len(x.shape)==1
        assert x.shape[0]%2 == 0
        n = x.shape[0] // 2
        assert x.shape == (2*n,)
        F = cls.bilinear_form(n)
        Fx = dot2(F.A, x)
        A = zeros2(2*n, 2*n)
        for i in range(2*n):
            u = array2([0]*(2*n))
            u[i] = 1
            v = dot2(u, Fx)
            u += v*x
            A[:, i] = u
        A %= 2
        A = Symplectic(A)
        return A

    def check(M):
        n = M.n
        A = Symplectic.bilinear_form(n)
        assert M.transpose() * A * M == A


def get_gen(n, pairs=None):
    #gen = [Symplectic.hadamard(n, i) for i in range(n)]
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
        gen.append(Symplectic.cnot(n, i, j))
        names.append("CN(%d,%d)"%(i,j))

    return gen, names


def get_encoder(source, target):
    assert isinstance(source, CSSCode)
    assert isinstance(target, CSSCode)
    src = Symplectic(source.to_symplectic())
    src_inv = src.inverse()
    tgt = Symplectic(target.to_symplectic())
    A = (src_inv * tgt).transpose()
    return A



def test_symplectic():

    n = 3
    I = Symplectic.identity(n)
    for idx in range(n):
      for jdx in range(n):
        if idx==jdx:
            continue
        CN_01 = Symplectic.cnot(n, idx, jdx)
        CN_10 = Symplectic.cnot(n, jdx, idx)
        assert CN_01*CN_01 == I
        assert CN_10*CN_10 == I
        lhs = CN_10 * CN_01 * CN_10
        rhs = Symplectic.swap(n, idx, jdx)
        assert lhs == rhs
        lhs = CN_01 * CN_10 * CN_01
        assert lhs == rhs

    #print(Symplectic.cnot(3, 0, 2))

    #if 0:
    cnot = Symplectic.cnot
    hadamard = Symplectic.hadamard
    n = 2
    gen = [cnot(n, 0, 1), cnot(n, 1, 0), hadamard(n, 0), hadamard(n, 1)]
    for A in gen:
        A.check()
    assert len(mulclose_fast(gen))==72 # index 10 in Sp(2, 4)

    n = 3
    gen = [
        cnot(n, 0, 1), cnot(n, 1, 0),
        cnot(n, 0, 2), cnot(n, 2, 0),
        cnot(n, 1, 2), cnot(n, 2, 1),
        hadamard(n, 0),
        hadamard(n, 1),
        hadamard(n, 2),
    ]
    for A in gen:
        A.check()
    assert len(mulclose_fast(gen))==40320 # index 36 in Sp(2,4)


    if 0:
        n = 2
        count = 0
        for A in enum2(4*n*n):
            A.shape = (2*n, 2*n)
            A = Symplectic(A)
            try:
                A.check()
                count += 1
            except:
                pass
        print(count) # 720 = |Sp(2, 4)|
        return

    for n in [1, 2]:
        gen = []
        for x in enum2(2*n):
            A = Symplectic.transvect(x)
            A.check()
            gen.append(A)
        assert len(mulclose_fast(gen)) == [6, 720][n-1]

    #return


    n = 4
    I = Symplectic.identity(n)
    H = Symplectic.hadamard(n, 0)
    assert H*H == I

    CN_01 = Symplectic.cnot(n, 0, 1)
    assert CN_01*CN_01 == I

    n = 3
    trivial = CSSCode(
        Lx=parse("1.."), Lz=parse("1.."), Hx=zeros2(0, n), Hz=parse(".1. ..1"))

    assert trivial.row_equal(CSSCode.get_trivial(3, 0))

    repitition = CSSCode(
        Lx=parse("111"), Lz=parse("1.."), Hx=zeros2(0, n), Hz=parse("11. .11"))

    assert not trivial.row_equal(repitition)

    CN_01 = Symplectic.cnot(n, 0, 1)
    CN_12 = Symplectic.cnot(n, 1, 2)
    CN_21 = Symplectic.cnot(n, 2, 1)
    CN_10 = Symplectic.cnot(n, 1, 0)
    encode = CN_12 * CN_01

    code = CN_01 ( trivial )
    assert not code.row_equal(repitition)
    code = CN_12 ( code )
    assert code.row_equal(repitition)

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
    



if __name__ == "__main__":

    name = argv.next()

    if argv.profile:
        import cProfile as profile
        profile.run("%s()"%name)

    elif name:

        fn = eval(name)
        fn()

    else:

        test_symplectic()


