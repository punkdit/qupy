#!/usr/bin/env python3

from collections import namedtuple
from functools import reduce
from operator import mul
from random import shuffle

import numpy
from numpy import concatenate as cat

from qupy.util import mulclose_fast
from qupy.tool import cross, choose
from qupy.argv import argv
from qupy.smap import SMap
from qupy.ldpc.solve import zeros2, shortstr, solve, dot2, array2, eq2, parse, pseudo_inverse, identity2
from qupy.ldpc.solve import enum2, row_reduce
from qupy.ldpc.css import CSSCode
from qupy.ldpc.decoder import StarDynamicDistance


def mulclose_find(gen, names, target, verbose=False, maxsize=None):
    gen = list(gen)
    ops = set(gen)
    lookup = dict((g, (names[i],)) for (i, g) in enumerate(gen))
    bdy = list(gen)
    dist = 1
    while bdy:
        _bdy = []
        shuffle(bdy)
        for g in bdy:
            shuffle(gen)
            for h in gen:
                k = g*h
                if k in ops:
                    if len(lookup[g]+lookup[h]) < len(lookup[k]):
                        lookup[k] = lookup[g]+lookup[h]
                        assert 0
                else:
                    word = lookup[g]+lookup[h]
                    if len(word) > dist:
                        dist = len(word)
                        if verbose:
                            print("dist:", dist)
                    lookup[k] = word
                    ops.add(k)
                    _bdy.append(k)

                if k==target:
                    return lookup[g]+lookup[h]
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



def get_cell(row, col, p=2):
    """
        return all matrices in bruhat cell at (row, col)
        These have shape (col, col+row).
    """

    if col == 0:
        yield zeros2(0, row)
        return

    if row == 0:
        yield identity2(col)
        return

    # recursive steps:
    m, n = col, col+row
    for left in get_cell(row, col-1, p):
        A = zeros2(m, n)
        A[:m-1, :n-1] = left
        A[m-1, n-1] = 1
        yield A

    els = list(range(p))
    vecs = list(cross((els,)*m))
    for right in get_cell(row-1, col, p):
        for v in vecs:
            A = zeros2(m, n)
            A[:, :n-1] = right
            A[:, n-1] = v
            yield A



def all_codes(m, n, q=2):
    """
        All full-rank generator matrices of shape (m, n)
    """
    assert m<=n
    col = m
    row = n-m
    return get_cell(row, col, q)




class Matrix(object):
    def __init__(self, A):
        self.A = A
        m, n = A.shape
        self.shape = A.shape
        assert n%2 == 0
        self.n = n
        self.m = m
        self.key = A.tostring() # careful !!

    def __str__(self):
        #s = str(self.A)
        #s = s.replace("0", ".")
        s = shortstr(self.A)
        return s

    def __mul__(self, other):
        assert isinstance(other, Matrix)
        assert other.m == self.n
        A = dot2(self.A, other.A)
        return Matrix(A)

    def __getitem__(self, key):
        return self.A[key]

    def __call__(self, other):
        assert isinstance(other, CSSCode)
        assert other.n*2 == self.n
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
        n = self.n//2
        Lx, Lz = LxLz[:, :n], LxLz[:, n:]
        Hx, Tz = HxTz[:, :n], HxTz[:, n:]
        Tx, Hz = TxHz[:, :n], TxHz[:, n:]
        code = CSSCode(Lx=Lx, Lz=Lz, Hx=Hx, Tz=Tz, Hz=Hz, Tx=Tx)
        return code

    def transpose(self):
        A = self.A.transpose().copy()
        return Matrix(A)

    def inverse(self):
        A = pseudo_inverse(self.A)
        return Matrix(A)

    def __eq__(self, other):
        assert isinstance(other, Matrix)
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
        return Matrix(A)

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
        return Matrix(A)

    @classmethod
    def cnot(cls, n, src, tgt):
        A = cls.identity(n).A
        assert src!=tgt
        A[tgt, src] = 1
        A[src+n, tgt+n] = 1
        return Matrix(A)

    @classmethod
    def sgate(cls, n, i):
        A = cls.identity(n).A
        assert 0<=i<n
        A[i+n, i] = 1
        return Matrix(A)
    
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
        return Matrix(A)

    @classmethod
    def symplectic_form(cls, n):
        A = zeros2(2*n, 2*n)
        I = identity2(n)
        A[:n, n:] = I
        A[n:, :n] = I
        A = Matrix(A)
        return A

    @classmethod
    def transvect(cls, x):
        assert len(x.shape)==1
        assert x.shape[0]%2 == 0
        n = x.shape[0] // 2
        assert x.shape == (2*n,)
        F = cls.symplectic_form(n)
        Fx = dot2(F.A, x)
        A = zeros2(2*n, 2*n)
        for i in range(2*n):
            u = array2([0]*(2*n))
            u[i] = 1
            v = dot2(u, Fx)
            u += v*x
            A[:, i] = u
        A %= 2
        A = Matrix(A)
        return A

    def is_symplectic(M):
        assert M.n % 2 == 0
        n = M.n//2
        A = Matrix.symplectic_form(n)
        return M.transpose() * A * M == A

    def normal_form(self):
        A = self.A
        #print("normal_form")
        #print(A)
        A = row_reduce(A)
        #print(A)
        m, n = A.shape
        j = 0
        for i in range(m):
            while A[i, j] == 0:
                j += 1
            i0 = i-1
            while i0>=0:
                r = A[i0, j]
                if r!=0:
                    A[i0, :] += A[i, :]
                    A %= 2
                i0 -= 1
            j += 1
        #print(A)
        A = Matrix(A)
        return A



def get_gen(n, pairs=None):
    gen = [Matrix.hadamard(n, i) for i in range(n)]
    names = ["H_%d"%i for i in range(n)]
    gen += [Matrix.sgate(n, i) for i in range(n)]
    names += ["S_%d"%i for i in range(n)]
    if pairs is None:
        pairs = []
        for i in range(n):
          for j in range(n):
            if i!=j:
                pairs.append((i, j))
    for (i, j) in pairs:
        assert i!=j
        gen.append(Matrix.cnot(n, i, j))
        names.append("CN(%d,%d)"%(i,j))

    return gen, names


def get_encoder(source, target):
    assert isinstance(source, CSSCode)
    assert isinstance(target, CSSCode)
    src = Matrix(source.to_symplectic())
    src_inv = src.inverse()
    tgt = Matrix(target.to_symplectic())
    A = (src_inv * tgt).transpose()
    return A



def test_symplectic():

    n = 3
    I = Matrix.identity(n)
    for idx in range(n):
      for jdx in range(n):
        if idx==jdx:
            continue
        CN_01 = Matrix.cnot(n, idx, jdx)
        CN_10 = Matrix.cnot(n, jdx, idx)
        assert CN_01*CN_01 == I
        assert CN_10*CN_10 == I
        lhs = CN_10 * CN_01 * CN_10
        rhs = Matrix.swap(n, idx, jdx)
        assert lhs == rhs
        lhs = CN_01 * CN_10 * CN_01
        assert lhs == rhs

    #print(Matrix.cnot(3, 0, 2))

    #if 0:
    cnot = Matrix.cnot
    hadamard = Matrix.hadamard
    n = 2
    gen = [cnot(n, 0, 1), cnot(n, 1, 0), hadamard(n, 0), hadamard(n, 1)]
    for A in gen:
        assert A.is_symplectic()
    Cliff2 = mulclose_fast(gen)
    assert len(Cliff2)==72 # index 10 in Sp(2, 4)

    CZ = array2([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1]])
    CZ = Matrix(CZ)
    assert CZ.is_symplectic()
    assert CZ in Cliff2

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
        assert A.is_symplectic()
    assert len(mulclose_fast(gen))==40320 # index 36 in Sp(2,4)

    if 0:
        # cnot's generate GL(2, n)
        n = 4
        gen = []
        for i in range(n):
          for j in range(n):
            if i!=j:
                gen.append(cnot(n, i, j))
        assert len(mulclose_fast(gen)) == 20160


    if 0:
        n = 2
        count = 0
        for A in enum2(4*n*n):
            A.shape = (2*n, 2*n)
            A = Matrix(A)
            try:
                assert A.is_symplectic()
                count += 1
            except:
                pass
        print(count) # 720 = |Sp(2, 4)|
        return

    for n in [1, 2]:
        gen = []
        for x in enum2(2*n):
            A = Matrix.transvect(x)
            assert A.is_symplectic()
            gen.append(A)
        G = mulclose_fast(gen)
        assert len(G) == [6, 720][n-1]

    n = 2
    Sp = G
    #print(len(Sp))
    found = set()
    for g in Sp:
        A = g.A.copy()
        A[:n, n:] = 0
        A[n:, :n] = 0
        found.add(str(A))
    #print(len(A))

    #return


    n = 4
    I = Matrix.identity(n)
    H = Matrix.hadamard(n, 0)
    assert H*H == I

    CN_01 = Matrix.cnot(n, 0, 1)
    assert CN_01*CN_01 == I

    n = 3
    trivial = CSSCode(
        Lx=parse("1.."), Lz=parse("1.."), Hx=zeros2(0, n), Hz=parse(".1. ..1"))

    assert trivial.row_equal(CSSCode.get_trivial(3, 0))

    repitition = CSSCode(
        Lx=parse("111"), Lz=parse("1.."), Hx=zeros2(0, n), Hz=parse("11. .11"))

    assert not trivial.row_equal(repitition)

    CN_01 = Matrix.cnot(n, 0, 1)
    CN_12 = Matrix.cnot(n, 1, 2)
    CN_21 = Matrix.cnot(n, 2, 1)
    CN_10 = Matrix.cnot(n, 1, 0)
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
    

def get_transvect(n):
    gen = []
    for x in enum2(2*n):
        A = Matrix.transvect(x)
        assert A.is_symplectic()
        gen.append(A)
    return gen


def test_isotropic():

    n = 2
    gen, _ = get_gen(n)
    print(len(mulclose_fast(gen)))
    return

    n = argv.get("n", 3)
    F = Matrix.symplectic_form(n).A

    found = []
    for A in all_codes(n, 2*n):
        B = dot2(dot2(A, F), A.transpose())
        if B.sum() == 0:
            A = Matrix(A)
            found.append(A)
            #print(A)

    found = set(found)
    print(len(found))

    gen, _ = get_gen(n)
    #gen = get_transvect(n)

    orbit = set()
    A = iter(found).__next__()

    bdy = [A]
    orbit = set(bdy)
    while bdy:
        _bdy = []
        for A in bdy:
          print(A)
          for g in gen:
            B = A*g
            print(B)
            print()
            B = B.normal_form()
            print(B)
            print()
            assert B in found
            if B not in orbit:
                _bdy.append(B)
                orbit.add(B)
        bdy = _bdy

    print(len(orbit))


def test_encode():

    n = 5

    target = parse("""
    1.11......
    .11.1.....
    .....111..
    .......111
    1.1.1.....
    ........1.
    ......1...
    1.........
    1.1.......
    ......1..1
    """)
    target = Matrix(target)
    assert target.is_symplectic()

    source = parse("""
    ...1......
    .1........
    .....1....
    .......1..
    ....1.....
    ........1.
    ......1...
    1.........
    ..1.......
    .........1
    """)
    source = Matrix(source)
    assert source.is_symplectic()

    def get_encoder(source, target):
        assert isinstance(source, CSSCode)
        assert isinstance(target, CSSCode)
        src = Matrix(source.to_symplectic())
        src_inv = src.inverse()
        tgt = Matrix(target.to_symplectic())
        A = (src_inv * tgt).transpose()
        return A

    #print(Matrix.cnot(2, 0, 1))
    #return

    #source = source.transpose()
    #target = target.transpose()

    def cnot(i, j):
        g = Matrix.cnot(n, i-1, j-1)
        g.name = "cnot(%d,%d)"%(i, j)
        return g

    assert cnot(3,1) == cnot(1,3).transpose()
    #gen = [cnot(3,1), cnot(2,3), cnot(2,5), cnot(4,3), cnot(5,3)]
    gen = [cnot(3,1), cnot(2,3), cnot(2,5), cnot(4,3), cnot(4,1), cnot(5,3), cnot(5,4),
        cnot(2,1),
    ]
    #gen = [cnot(i,j) for i in range(1,n+1) for j in range(1,n+1) if i!=j]
    names = [g.name for g in gen]

    gen = [g.transpose() for g in gen]

    #A = (source.inverse() * target).transpose()
    #A = source.inverse() * target
    A = target * source.inverse()
    assert A.is_symplectic()
    #print(A)

    words = set()
    for trial in range(100):
        word = mulclose_find(gen, names, A)
        if word is None:
            break
        if word not in words:
            print(word)
            words.add(word)

    if word is None:
        print("not found")
        return
    ops = [gen[names.index(c)] for c in word]
    op = reduce(mul, ops)
    assert op*source == target
    



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
        test_isotropic()


