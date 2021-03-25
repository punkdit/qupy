#!/usr/bin/env python3

"""
Construct transversal S gate on a folded surface code.
See: https://arxiv.org/abs/1603.02286
"""

import math
from functools import reduce
from operator import mul, matmul

import numpy
import numba as nb

from qupy.argv import argv
if argv.complex64:
    from qupy import scalar
    scalar.scalar = numpy.complex64
    scalar.EPSILON = 1e-6
from qupy.abstract import Space
from qupy.dense import Qu, Gate, Vector, EPSILON, scalar
from qupy.dense import genidx, bits, is_close, on, off, scalar
from qupy.dense import commutator, anticommutator
from qupy.test import test

r2 = math.sqrt(2)


I, X, Z, S, T = Gate.I, Gate.X, Gate.Z, Gate.S, Gate.T

assert X*X == I
assert Z*Z == I
assert S*S == Z
#assert (T*T).is_close(S, 1e-6)
assert T*T == S

Sd = S.dag()
assert S*Sd == I


class Operator(object):
    def __init__(self, n, d=2):
        self.n = n
        self.d = d
        self.shape = (d,)*n
        self.dtype = scalar

    def __str__(self):
        return "%s(%d)"%(self.__class__.__name__, self.n)

    def __call__(self, v, u):
        assert 0, "abstract base class"

    def todense(self):
        assert self.d ** self.n <= 2**13
        shape = self.shape + self.shape
        A = numpy.zeros(shape, dtype=scalar)
        u = numpy.zeros(self.shape, dtype=scalar)
        jdx = (slice(None),)*self.n
        for idx in numpy.ndindex(self.shape):
            u[idx] = 1.
            v = self(u)
            A[jdx+idx] = v
            u[idx] = 0.
        return A

    def __add__(self, other):
        return Addop(self, other)

    def __mul__(self, other):
        return Mulop(self, other)

    @classmethod    
    def make_I(cls, n):
        def func(v, u):
            u[:] = v
        return Numop(n, func)

    @classmethod    
    def make_zop(cls, n, idxs):
        stmt = """
        def func(v, u):
          for src in numpy.ndindex(v.shape):
            sign = 1
            #for i, tgt in enumerate(idxs):
            for tgt in %s:
                if src[tgt] == 1:
                    sign *= -1
            u[src] = sign*v[src]
        """ % idxs

        func = mkfunc(stmt)
        return Numop(n, func)

    @classmethod    
    def make_xop(cls, n, idxs):
        stmt = """
        def func(v, u):
          for src in numpy.ndindex(v.shape):
            #tgt = list(src)
            #for idx in %s:
            #    tgt[idx] = 1-tgt[idx]
            #tgt = tuple(tgt)
            #tgt = (0,)
            u[src] = v[src]
        """ % idxs

        print("make_xop")
        print(stmt)
        func = mkfunc(stmt)
        return Numop(n, func)

    @classmethod    
    def make_cz(cls, n):
        IDX = (1,)*n
        def func(v, u):
          for idx in numpy.ndindex(v.shape):
            if idx == IDX:
                u[idx] = -v[idx]
            else:
                u[idx] = v[idx]
        return Numop(n, func)


def mkfunc(stmt):
    i = 0
    while i < len(stmt) and stmt[i] == '\n':
        i += 1
    stmt = stmt[i:]
    assert stmt
    if stmt[0] == ' ':
        stmt = "if 1:\n"+stmt
    ns = dict(globals())
    exec(stmt, ns, ns)
    func = ns['func']
    return func


class Numop(Operator):
    def __init__(self, n, func, d=2):
        Operator.__init__(self, n, d)
        s = ','.join("n"*n)
        desc = "(%s)->(%s)"%(s, s)
        print("Numop:", desc)
        func = nb.guvectorize(desc)(func)
        self.func = func

    def __call__(self, u, v=None):
        assert u.shape == self.shape
        assert u.dtype == self.dtype
        if v is None:
            v = numpy.zeros(self.shape, dtype=self.dtype)
        self.func(u, v)
        return v


class Addop(Operator):
    def __init__(self, lhs, rhs):
        assert lhs.n == rhs.n
        Operator.__init__(self, lhs.n)
        self.lhs = lhs
        self.rhs = rhs

    def __call__(self, u):
        lhs = self.lhs(u)
        rhs = self.rhs(u)
        return lhs + rhs


class Mulop(Addop):
    def __call__(self, u):
        u = self.rhs(u)
        u = self.lhs(u)
        return u




def test_op(spec):
    n = len(spec)
    zidxs = [i for i in range(n) if spec[i]=='Z']
    xidxs = [i for i in range(n) if spec[i]=='X']
    zop = Operator.make_zop(n, zidxs)
    xop = Operator.make_xop(n, xidxs)
    lhs = zop * xop

    rhs = [I]*n
    for i in zidxs:
        rhs[i] = Z
    for i in xidxs:
        rhs[i] = X
    rhs = reduce(matmul, rhs)
    
    shape = (2,)*n
    qu = Qu(shape, 'u'*n)
    #u = numpy.zeros(shape, dtype=scalar)
    u = qu.v
    for idx in numpy.ndindex(shape):
        u[idx] = 1.
        v_lhs = lhs(u)
        v_rhs = rhs*qu
        assert numpy.allclose(v_lhs, v_rhs.v)
        u[idx] = 0.


def main():

    test_op("IX")

    #test_op("ZZ")
    #test_op("ZIII")

    return

    ZZ = Operator.make_zop(2, [0,1])
    lhs = ZZ.todense()
    rhs = Z@Z
    print(lhs)
    print(rhs.shortstr())
    print(rhs)
    print(rhs.shape)
    assert numpy.allclose(lhs, rhs.v)

    ZIZ = Operator.make_zop(3, [0,2])
    lhs = ZIZ.todense()
    rhs = (Z@I@Z).v
    print(lhs)
    print(rhs)
    assert numpy.allclose(lhs, rhs)

    n = argv.get("n", 5)
    op = Operator.make_cz(n)
    v = numpy.zeros((2,)*n, dtype=scalar)
    v[:] = 1
    u = op(v)

    if argv.check:
        for idx in numpy.ndindex(v.shape):
            if idx == (1,)*n:
                assert u[idx] == -1
            else:
                assert u[idx] == +1

#    import time
#    while 1:
#        time.sleep(1)


if __name__ == "__main__":

    main()

    print("OK")

