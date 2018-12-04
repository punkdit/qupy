#!/usr/bin/env python3

#from math import sqrt
#from random import choice, randint, seed, shuffle

from functools import reduce
from operator import mul, matmul, add

import numpy

from qupy.argv import argv
from qupy.util import mulclose, mulclose_fast



class scalar(object):
    dtype = numpy.int64
    one = 1
    zero = 0


class Algebra(object):
    def __init__(self, dim, struct, names=None):
        self.dim = dim
        self.struct = numpy.array(struct)
        basis = []
        for i in range(dim):
            v = [scalar.zero]*dim
            v[i] = scalar.one
            op = Operator(self, v)
            basis.append(op)
        self.basis = basis
        if names is None:
            names = "IABCDEFGHJKLMNOPQRSTUVWXYZ"[:dim]
        assert len(names)==dim
        self.names = names

    def is_associative(self):
        for a in self.basis:
         for b in self.basis:
          for c in self.basis:
            if (a*b)*c != a*(b*c):
                #print(a, b, c)
                return False
        return True

    def parse(self, desc):
        n = len(desc)
        idxs = tuple(self.names.index(c) for c in desc)
        return Tensor(n, self, {idxs : scalar.one})

    def build_code(self, ops):
        return StabilizerCode(self, ops)


class Operator(object):
    def __init__(self, algebra, v):
        self.algebra = algebra
        assert len(v) == algebra.dim
        self.v = numpy.array(v)

    def __str__(self):
        algebra = self.algebra
        names = algebra.names
        v = self.v
        s = []
        for i in range(algebra.dim):
            if v[i]==0:
                continue
            elif v[i]==1:
                term = "%s"%(names[i])
            elif v[i]==-1:
                term = "-%s"%(names[i])
            else:
                term = "%s*%s"%(v[i], names[i])
            s.append(term)
        s = '+'.join(s)
        s = s.replace("+-", "-")
        s = s or "0"
        return s

    def __repr__(self):
        return "Operator(%s)"%(self.v,)

    def __eq__(self, other):
        assert other.algebra is self.algebra
        return numpy.allclose(self.v, other.v)

    def __ne__(self, other):
        assert other.algebra is self.algebra
        return not numpy.allclose(self.v, other.v)

    def __add__(self, other):
        assert other.algebra is self.algebra
        v = self.v + other.v
        return Operator(self.algebra, v)

    def __sub__(self, other):
        assert other.algebra is self.algebra
        v = self.v - other.v
        return Operator(self.algebra, v)

    def __neg__(self):
        v = -self.v
        return Operator(self.algebra, v)

    def __rmul__(self, r):
        v = r*self.v
        return Operator(self.algebra, v)

    def __mul__(self, other):
        algebra = self.algebra
        assert algebra is self.algebra
        dim = algebra.dim
        v = [scalar.zero] * dim
        struct = algebra.struct
        for i in range(dim):
          for j in range(dim):
            r = self.v[i] * other.v[j]
            if r:
                v += r*struct[i, j]
        return Operator(self.algebra, v)

    def __matmul__(self, other):
        algebra = self.algebra
        assert algebra is self.algebra
        self = Tensor.promote(self)
        return self.__matmul__(other)


class Tensor(object):
    "operator that lives in the n-fold tensor product algebra"
    def __init__(self, n, algebra, coefs):
        self.n = n
        self.algebra = algebra
        #self.coefs = coefs # map tuple of idxs -> value
        self.coefs = dict((k, v) for (k, v) in coefs.items() if v!=scalar.zero)
        self._str = None

    def __str__(self):
        #return "Tensor(%s)"%(self.coefs,)
        if self._str is not None:
            return self._str
        algebra = self.algebra
        keys = list(self.coefs.keys())
        keys.sort()
        terms = []
        for idx in keys:
            val = self.coefs[idx]
            assert val != scalar.zero
            name = ''.join(algebra.names[i] for i in idx)
            if val == 1:
                term = name
            elif val == -1:
                term = "-"+name
            else:
                term = "%s*%s"%(val, name)
            terms.append(term)
        s = '+'.join(terms)
        s = s.replace("+-", "-")
        s = s or "0"
        self._str = s
        return s

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        assert self.algebra is other.algebra
        assert self.n == other.n
        return self.coefs == other.coefs

    def __ne__(self, other):
        assert self.algebra is other.algebra
        assert self.n == other.n
        return self.coefs != other.coefs

    @classmethod
    def promote(cls, item):
        if isinstance(item, Operator):
            algebra = item.algebra
            coefs = {}
            for i in range(algebra.dim):
                if item.v[i] != scalar.zero:
                    coefs[(i,)] = item.v[i]
            return Tensor(1, algebra, coefs)
        if isinstance(item, Tensor):
            return item
        raise Exception(str(item))

    def __matmul__(self, other):
        other = Tensor.promote(other)
        assert self.algebra is other.algebra
        n = self.n + other.n
        coefs = {}
        for idx, val in self.coefs.items():
          for jdx, wal in other.coefs.items():
            key = idx+jdx
            assert key not in coefs # <-- remove me
            coefs[key] = val * wal
        return Tensor(n, self.algebra, coefs)

    def __add__(self, other):
        other = Tensor.promote(other)
        assert self.algebra is other.algebra
        assert self.n == other.n
        coefs = dict(self.coefs)
        zero = scalar.zero
        for key, value in other.coefs.items():
            coefs[key] = coefs.get(key, zero) + value # test for zero?
        return Tensor(self.n, self.algebra, coefs)

    def __sub__(self, other):
        other = Tensor.promote(other)
        assert self.algebra is other.algebra
        assert self.n == other.n
        coefs = dict(self.coefs)
        zero = scalar.zero
        for key, value in other.coefs.items():
            coefs[key] = coefs.get(key, zero) - value # test for zero?
        return Tensor(self.n, self.algebra, coefs)

    def __rmul__(self, r):
        coefs = dict((key, r*v) for (key, v) in self.coefs.items())
        return Tensor(self.n, self.algebra, coefs)

    def __mul__(self, other):
        # slow but it works...
        assert self.algebra is other.algebra
        assert self.n == other.n
        zero = scalar.zero
        algebra = self.algebra
        struct = algebra.struct
        dim = algebra.dim
        n = self.n
        A = Tensor(self.n, self.algebra, {})
        for idx, val in self.coefs.items():
          for jdx, wal in other.coefs.items():
            r = val*wal
            if r==scalar.zero:
                continue
            tens_ops = []
            for i in range(n):
                coefs = {}
                v = struct[idx[i], jdx[i]]
                v = Operator(algebra, v)
                tens_ops.append(v)
            v = reduce(matmul, tens_ops)
            v = r * v
            A = A+v

        return A



#"""
#class Lazy(object):
#
#
#class LazyMul(Lazy):
#    def __init__(self, left, right):
#        assert isinstance(left, LazyOp)
#        assert isinstance(right, LazyOp)
#
#
#class LazyOp(Lazy):
#    def __init__(self, name):
#        self.name = name
#
#    def __mul__(self, other):
#
#"""



def build_algebra(names, rel):
    names = list(names)
    assert names[0] == "I" # identity
    dim = len(names)
    struct = {}
    struct[0, 0, 0] = scalar.one
    for i in range(1, dim):
        struct[0, i, i] = scalar.one
        struct[i, 0, i] = scalar.one

    eqs = rel.split()
    for eq in eqs:
        lhs, rhs = eq.split("=")
        A, B = lhs.split("*")
        i = names.index(A)
        j = names.index(B)
        rhs, C = rhs[:-1], rhs[-1]
        k = names.index(C)
        if not rhs:
            val = scalar.one
        elif rhs == "-":
            val = -scalar.one
        else:
            assert 0, repr(eq)
        assert struct.get((i, j, k)) is None
        struct[i, j, k] = val

    A = numpy.zeros((dim, dim, dim), dtype=scalar.dtype)
    for key, value in struct.items():
        A[key] = value
    algebra = Algebra(dim, A, names)
    return algebra


class StabilizerCode(object):
    def __init__(self, algebra, ops):
        ops = ops.split()
        self.n = len(ops[0])
        ops = [algebra.parse(s) for s in ops]
        for g in ops:
          for h in ops:
            assert g*h == h*g 
        self.ops = list(ops)

    def get_projector(self):
        G = mulclose_fast(self.ops)

        # build projector onto codespace
        #P = (1./len(G))*reduce(add, G)
        P = reduce(add, G)
        return P

    


def test():

    algebra = build_algebra("IXZY",
        "X*X=I Z*Z=I Y*Y=-I X*Z=Y Z*X=-Y X*Y=Z Y*X=-Z Z*Y=-X Y*Z=X")

    I, X, Z, Y = algebra.basis
    assert X*Z == -Z*X
    assert I*X == X
    assert X*X == I
    assert I*X*X == I
    assert I*(X*X) == I
    assert X*Z == Y

    assert algebra.is_associative()

    assert (X + Z) * (X + Z) == 2*I

    II = I@I
    assert II * II == II
    
    items = list(algebra.basis)
    items.append(X+Z)

    items = [Tensor.promote(a) for a in items]

    _H = Tensor.promote(X + Z)
    _I = Tensor.promote(I)
    assert _H*_H == 2*_I

    for A in items:
      for B in items:
        for C in items:
          for D in items:
            assert (A@B) * (C@D) == (A*C) @ (B*D)

    assert str(X@X) == "XX"
    assert algebra.parse("XX") == X@X

    # five qubit code ----------------

    code = algebra.build_code('XZZXI IXZZX XIXZZ ZXIXZ')
    P = code.get_projector()
    assert P*P == 16*P

    # steane code ----------------

    code = algebra.build_code("XXXXIII XXIIXXI XIXIXIX ZZZZIII ZZIIZZI ZIZIZIZ")
    P = code.get_projector()

    assert P*P == 64*P





if __name__ == "__main__":

    name = argv.next() or "test"

    fn = eval(name)
    fn()


