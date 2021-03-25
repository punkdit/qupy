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

DEBUG = argv.debug

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

    def __call__(self, v, u=None):
        u = v.copy()
        return u

    def __eq__(lhs, rhs):
        assert lhs.n == rhs.n
        #return numpy.allclose(lhs.todense(), rhs.todense())
        n = lhs.n
        N = 2**n
        u = numpy.zeros(N, dtype=scalar)
        u1 = u.view()
        u1.shape = (2,)*n
        for i in range(N):
            u[i] = 1.
            lhs1 = lhs(u1)
            rhs1 = rhs(u1)
            if not numpy.allclose(lhs1, rhs1):
                return False
            u[i] = 0.
        return True

    def __ne__(lhs, rhs):
        return not lhs.__eq__(rhs)

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
        return AddOp(self, other)

    def __sub__(self, other):
        return SubOp(self, other)

    def __mul__(self, other):
        return MulOp(self, other)

    def __rmul__(self, other):
        return RMulOp(self, other)

    def __neg__(self):
        return RMulOp(self, -1.)

    @classmethod    
    def make_I(cls, n):
        return Operator(n)
        def func(v, u):
            u[:] = v
        return Numop(n, func)

#    @classmethod    
#    def make_zop(cls, n, idxs):
#        stmt = """
#        def func(v, u):
#          for src in numpy.ndindex(v.shape):
#            sign = 1
#            #for i, tgt in enumerate(idxs):
#            for tgt in %s:
#                if src[tgt] == 1:
#                    sign *= -1
#            u[src] = sign*v[src]
#        """ % idxs
#
#        func = mkfunc(stmt)
#        return Numop(n, func)

    @classmethod    
    def make_zop(cls, n, idxs):
        if DEBUG:print("make_zop", n, idxs)
        stmt = """
        def func(v, u):
          for src in range(%d):
            sign = 1
            for tgt in %s:
                if src & tgt:
                    sign *= -1
            u[src] = sign*v[src]
        """ % (2**n, [2**i for i in idxs])

        func = mkfunc(stmt)
        return Numop(n, func)

    @classmethod    
    def make_xop(cls, n, idxs):
        if DEBUG:print("make_xop", n, idxs)
        stmt = """
        def func(v, u):
          for src in range(%d):
            tgt = src
            for i in %s:
                tgt ^= i
            u[tgt] = v[src]
        """ % (2**n, [2**i for i in idxs])

        func = mkfunc(stmt)
        return Numop(n, func)

    @classmethod
    def make_op(cls, spec):
        n = len(spec)
        zidxs = [i for i in range(n) if spec[i]=='Z']
        xidxs = [i for i in range(n) if spec[i]=='X']
        op = Operator(n)
        if zidxs:
            op = Operator.make_zop(n, zidxs)
        if xidxs:
            xop = Operator.make_xop(n, xidxs)
            op = op * xop
        return op

    @classmethod    
    def make_cz(cls, n):
        if DEBUG:print("make_cz", n)
        stmt = """
        def func(v, u):
            u[:] = v[:]
            u[%d] = -v[%d]
        """ % (2**n-1, 2**n-1)
        func = mkfunc(stmt)
        return Numop(n, func)



def mkfunc(stmt):
    if DEBUG:print("mkfunc")
    if DEBUG:print(stmt)
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
        #s = ','.join("n"*n)
        #desc = "(%s)->(%s)"%(s, s)
        desc = "(n)->(n)"
        if DEBUG:print("Numop:", desc)
        func = nb.guvectorize(desc)(func)
        self.func = func

    def __call__(self, u, v=None):
        assert u.shape == self.shape
        assert u.dtype == self.dtype
        u = u.view()
        shape = (2**self.n,)
        u.shape = shape
        if v is None:
            v = numpy.zeros(shape, dtype=self.dtype)
        self.func(u, v)
        v.shape = self.shape
        return v


class AddOp(Operator):
    def __init__(self, lhs, rhs):
        assert lhs.n == rhs.n
        Operator.__init__(self, lhs.n)
        self.lhs = lhs
        self.rhs = rhs

    def __call__(self, u):
        lhs = self.lhs(u)
        rhs = self.rhs(u)
        return lhs + rhs


class SubOp(AddOp):
    def __call__(self, u):
        lhs = self.lhs(u)
        rhs = self.rhs(u)
        return lhs - rhs



class MulOp(AddOp):
    def __call__(self, u):
        u = self.rhs(u)
        u = self.lhs(u)
        return u

class RMulOp(Operator):
    def __init__(self, lhs, r):
        Operator.__init__(self, lhs.n)
        self.lhs = lhs
        self.r = r

    def __call__(self, u):
        u = self.lhs(u)
        u = self.r * u
        return u





def test_op(spec):
    n = len(spec)
    zidxs = [i for i in range(n) if spec[i]=='Z']
    xidxs = [i for i in range(n) if spec[i]=='X']
    zop = Operator.make_zop(n, zidxs)
    xop = Operator.make_xop(n, xidxs)
    lhs = zop * xop
    print("lhs:")
    print(lhs.todense())

    rhs = [I]*n
    for i in zidxs:
        rhs[i] = Z
    for i in xidxs:
        rhs[i] = X
    #rhs = list(reversed(rhs))
    rhs = reduce(matmul, rhs)
    print(rhs.v)

    #assert numpy.allclose(lhs.todense(), rhs.v)
    #return
    
    shape = (2,)*n
    qu = Qu(shape, 'u'*n)
    #u = numpy.zeros(shape, dtype=scalar)
    u = qu.v
    for idx in numpy.ndindex(shape):
        u[idx] = 1.
        v_lhs = lhs(u)
        v_rhs = rhs*qu
        print(v_lhs)
        print(v_rhs.v)
        assert numpy.allclose(v_lhs, v_rhs.v)
        u[idx] = 0.


def main():

    ZI = Operator.make_op("ZI")
    assert ZI == ZI

    II = Operator.make_op("II")
    assert II==II

    XI = Operator.make_op("XI")
    IZ = Operator.make_op("IZ")
    IX = Operator.make_op("IX")
    ZZ = Operator.make_op("ZZ")
    XX = Operator.make_op("XX")
    assert ZZ*ZZ == II
    assert ZZ*XX == XX*ZZ
    assert IZ*XI == XI*IZ
    assert ZI != XI
    assert II - II == 0.*II
    for op in [ZI, IZ, XI, IX, ZZ, XX]:
        assert op*op == II
        assert op - 2*op == -op
    assert ZI*XI != XI*ZI
    assert ZI*XI == -(XI*ZI) 

    spec = "ZXIZXIZIXIZ"
    op = Operator.make_op(spec)
    assert op*op == Operator.make_I(op.n)

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

