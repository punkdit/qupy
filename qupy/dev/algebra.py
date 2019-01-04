#!/usr/bin/env python3

import numpy

from qupy.tool import fstr


class scalar(object):
    zero = 0.0
    one = 1.0
    dtype = numpy.complex128

EPSILON = 1e-8



class Algebra(object):
    def __init__(self, dim, names=None, struct=None):
        self.dim = dim 
        if names is None:
            names = "IABCDEFGHJKLMNOPQRSTUVWXYZ"[:dim]
        assert len(names)==dim
        self.names = names

        if struct is not None:
            struct = numpy.array(struct)
            self.struct = struct
            self.build_lookup()
        basis = []
        for i in range(dim):
            op = Tensor({(i,):scalar.one}, 1, self)
            basis.append(op)
        self.basis = basis

    def build_lookup(self):
        lookup = {}
        dim = self.dim
        struct = self.struct
        for i in range(dim):
          for j in range(dim):
            v = struct[i, j]
            coefs = {}
            for k in range(dim):
                if abs(v[k])>EPSILON:
                    coefs[(k,)] = v[k]
            v = Tensor(coefs, 1, self)
            lookup[i, j] = v
        self.lookup = lookup

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
        return Tensor({idxs : scalar.one}, n, self)

    def __getattr__(self, attr):
        if attr in self.names:
            return self.parse(attr)
        raise AttributeError

    def get_zero(self, grade):
        op = Tensor({}, grade, self)
        return op

    def construct(self, cs):
        return Tensor(cs, algebra=self)


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
    algebra = Algebra(dim, names, A)
    return algebra


class Tensor(object):

    """ Some kind of graded ring element... I*I*I + X*X*X etc.
        There is no real reason to make this homogeneous,
        but i do for now.
    """

    zero = 0.0
    one = 1.0
    def __init__(self, coefs, grade=None, algebra=None):
        # map key -> coeff, key is ("A", "B") etc.
        assert coefs or (grade is not None)
        if grade is None:
            key = iter(coefs.keys()).__next__()
            grade = len(key)
        self.coefs = dict(coefs)
        self.grade = grade
        self.algebra = algebra
        self.keys = list(self.coefs.keys()) # cache this
        self.items = list(self.coefs.items()) # cache this

    def permute(self, perm):
        coefs = {}
        idxs = list(range(self.grade))
        for (k, v) in self.items:
            k = tuple(k[perm[i]] for i in idxs)
            coefs[k] = v
        return Tensor(coefs, self.grade, self.algebra)

    def get_zero(self):
        return Tensor({}, self.grade, self.algebra)

    def get_keys(self):
        #return list(self.coefs.keys())
        return self.keys

    def __getitem__(self, key):
        return self.coefs.get(key, self.zero)

    def __add__(self, other):
        assert self.grade == other.grade # i guess this is not necessary...
        coefs = dict(self.coefs)
        for (k, v) in other.coefs.items():
            coefs[k] = coefs.get(k, self.zero) + v
        return Tensor(coefs, self.grade, self.algebra)

    def __sub__(self, other):
        assert self.grade == other.grade
        coefs = dict(self.coefs)
        for (k, v) in other.coefs.items():
            coefs[k] = coefs.get(k, self.zero) - v
        return Tensor(coefs, self.grade, self.algebra)

    def __matmul__(self, other):
        coefs = {} 
        for (k1, v1) in self.coefs.items():
          for (k2, v2) in other.coefs.items():
            k = k1+k2
            assert k not in coefs
            coefs[k] = v1*v2
        return Tensor(coefs, self.grade+other.grade, self.algebra)

    def __rmul__(self, r):
        coefs = {}
        for (k, v) in self.coefs.items():
            coefs[k] = complex(r)*v
        return Tensor(coefs, self.grade, self.algebra)

    def __neg__(self):
        coefs = {}
        for (k, v) in self.coefs.items():
            coefs[k] = -v
        return Tensor(coefs, self.grade, self.algebra)

    def __len__(self):
        return len(self.coefs)

    def subs(self, rename):
        the_op = Tensor({}, self.grade, self.algebra) # zero
        algebra = self.algebra
        one = self.one
        for (k, v) in self.coefs.items():
            final = None
            for ki in k:
                c = algebra.names[ki]
                op = rename.get(c, Tensor({(ki,) : one}, algebra=self.algebra))
                if final is None:
                    final = op
                else:
                    final = final @ op # tensor
            the_op = the_op + complex(v)*final # ARRGGGHHH !!
        return the_op

    def evaluate(self, rename):
        algebra = self.algebra
        the_op = None
        one = self.one
        for (k, v) in self.coefs.items():
            final = None
            for ki in k:
                c = algebra.names[ki]
                op = rename[c]
                if final is None:
                    final = op
                else:
                    final = final @ op # tensor
            the_op = complex(v)*final if the_op is None else the_op + complex(v)*final
        return the_op

    def __str__(self):
        ss = []
        algebra = self.algebra
        keys = list(self.coefs.keys())
        keys.sort()
        for k in keys:
            v = self.coefs[k]
            s = ''.join(algebra.names[ki] for ki in k)
            if abs(v) < EPSILON:
                continue
            elif abs(v-1) < EPSILON:
                pass
            elif abs(v+1) < EPSILON:
                s = "-"+s
            else:
                s = fstr(v)+"*"+s
            ss.append(s)
        ss = '+'.join(ss) or "0"
        ss = ss.replace("+-", "-")
        return ss

    def __repr__(self):
        return "Tensor(%s)"%(self.coefs)

    def norm(self):
        return sum(abs(val) for val in self.coefs.values())

    def __eq__(self, other):
        if not isinstance(other, Tensor) and other==0:
            other = self.get_zero()
        return (self-other).norm() < EPSILON

    def __ne__(self, other):
        if not isinstance(other, Tensor) and other==0:
            other = self.get_zero()
        return (self-other).norm() > EPSILON

    #def __hash__(self):
    #    return hash((str(self), self.grade))

    def __mul__(self, other): # HOTSPOT
        # slow but it works...
        assert self.algebra is not None
        assert self.algebra is other.algebra
        assert self.grade == other.grade
        zero = scalar.zero
        algebra = self.algebra
        struct = algebra.struct
        dim = algebra.dim
        n = self.grade
        coefs = {}
        for idx, val in self.items:
          if abs(val)<EPSILON:
              continue
          for jdx, wal in other.items:
            r = val*wal
            if abs(r)<EPSILON:
                continue
            key = ()
            for i in range(n):
                op = algebra.lookup[idx[i], jdx[i]]
                assert len(op.keys)==1
                k = op.keys[0]
                key = key + k
                r *= op.coefs[k]
            coefs[key] = coefs.get(key, 0.) + r

        A = Tensor(coefs, self.grade, algebra)
        return A

    @classmethod
    def tensor(cls, ops):
        coefs = {}
        _tensor_reduce(ops, 1., coefs)
        grade = reduce(add, [op.grade for op in ops])
        algebra = ops[0].algebra
        A = Tensor(coefs, grade, algebra)
        return A
        

def _tensor_reduce(ops, r, coefs):
    val = 1.
    key = ()
    for op in ops:
        k = op.keys[0]
        key = key + k
        val *= op.coefs[k]
    coefs[key] = coefs.get(key, 0.) + r*val


