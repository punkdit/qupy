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
        self.N = d**n
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
        if isinstance(other, Operator):
            return MulOp(self, other)
        return self.__call__(other)

    def __rmul__(self, other):
        return RMulOp(self, other)

    def __neg__(self):
        return RMulOp(self, -1.)

    @classmethod    
    def make_I(cls, n):
        return Operator(n)
        #def func(v, u):
        #    u[:] = v
        #return Numop(n, func)

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
    def make_phase_op(cls, n, idxs, phase):
        if DEBUG:print("make_phase_op", n, idxs, phase)
        stmt = """
        def func(v, u):
          for src in range(%d):
            phase = 1
            for tgt in %s:
                if src & tgt:
                    phase *= %s
            u[src] = phase*v[src]
        """ % (2**n, [2**i for i in idxs], phase)
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

    @classmethod
    def XXX_make_tensor(cls, ops):
        assert 0, "FAIL"
        n = len(ops)
        N = 2**n
        d = 2
        code = Code()
        code.append("def func(v, u):").indent()
        code.append("for j in range(%d):"%N).indent()
        bit = 1
        for op in ops:
            assert op.shape == (2, 2)
            for i in range(N):
                code.append("if j & %d == 0:"%bit).indent()
                code.dedent()
                code.append("else:").indent()
                code.dedent()
                code.append("u[%d] = r"%i)
            bit *= 2
    
        #code.append("u[j] = v[j]").indent()
        
        print("make_tensor:")
        print('\n'.join(code.lines))
        func = code.mkfunc()
        return Numop(n, func)

    @classmethod
    def make_tensor1(cls, n, A, idx):
        N = 2**n
        d = 2
        assert A.shape == (2,2)
        code = Code()
        code.append("def func(v, u):").indent()
        code.append("for j in range(%d):"%N).indent()
        code.append("if j & %d == 0:"%(2**idx)).indent()
        for i in range(2):
            r = A[i, 0]
            if r.real == r:
                r = r.real
            if abs(r) > EPSILON:
                index = "j" if i==0 else "j^%d"%(2**idx)
                code.append("u[%s] += %s*v[j]"%(index, r))
        code.dedent()
        code.append("else:").indent()
        for i in range(2):
            r = A[i, 1]
            if r.real == r:
                r = r.real
            if abs(r) > EPSILON:
                index = "j" if i==1 else "j^%d"%(2**idx)
                code.append("u[%s] += %s*v[j]"%(index, r))
        code.dedent()
        
        #print("make_tensor:")
        #print('\n'.join(code.lines))
        func = code.mkfunc()
        return Numop(n, func)

    @classmethod
    def make_tensor(cls, ops):
        n = len(ops)
        A = None
        for idx, op in enumerate(ops):
            #print("make_tensor", idx, op.shortstr())
            if op==Gate.I:
                #print("continue")
                continue
            B = cls.make_tensor1(n, op, idx)
            if A is None:
                A = B
            else:
                A = B*A
        return A

    @classmethod
    def make_control(cls, n, A, tgt, src):
        N = 2**n
        d = 2
        assert A.shape == (2,2)
        code = Code()
        code.append("def func(v, u):").indent()
        code.append("for j in range(%d):"%N).indent()
        code.append("if j & %d == 0:"%(2**src)).indent()
        code.append("u[j] += v[j]") # src bit is off
        code.dedent()
        code.append("else:").indent() # src bit is on
        code.append("if j & %d == 0:"%(2**tgt)).indent()
        for i in range(2):
            r = A[i, 0]
            if r.real == r:
                r = r.real
            if abs(r) > EPSILON:
                index = "j" if i==0 else "j^%d"%(2**tgt)
                code.append("u[%s] += %s*v[j]"%(index, r))
        code.dedent()
        code.append("else:").indent()
        for i in range(2):
            r = A[i, 1]
            if r.real == r:
                r = r.real
            if abs(r) > EPSILON:
                index = "j" if i==1 else "j^%d"%(2**tgt)
                code.append("u[%s] += %s*v[j]"%(index, r))
        code.dedent()
        code.dedent()
        
        print("make_control:")
        print('\n'.join(code.lines))
        func = code.mkfunc()
        return Numop(n, func)


class Code(object):
    INDENT = "    "
    def __init__(self):
        self._indent = 0
        self.lines = []
    def append(self, line):
        line = line.strip()
        indent = self.INDENT * self._indent
        self.lines.append(indent + line)
        return self
    def indent(self):
        self._indent += 1
        return self
    def dedent(self):
        self._indent -= 1
        return self
    def mkfunc(self):
        lines = '\n'.join(self.lines)
        return mkfunc(lines)


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
        assert u.dtype == self.dtype
        reshape = u.shape == self.shape
        shape = (2**self.n,)
        if reshape:
            u = u.view()
            u.shape = shape
        assert u.shape == (self.N,)
        if v is None:
            v = numpy.zeros(shape, dtype=self.dtype)
        self.func(u, v)
        if reshape:
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


def test_op():

    make_op = Operator.make_op

    I = make_op("I")
    Z = make_op("Z")
    assert Z==Z
    assert Z*Z==I

    ZI = make_op("ZI")
    assert ZI == ZI

    II = make_op("II")
    assert II==II

    XI = make_op("XI")
    IZ = make_op("IZ")
    IX = make_op("IX")
    ZZ = make_op("ZZ")
    XX = make_op("XX")
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

    assert II == Operator.make_tensor1(2, Gate.I, 1)
    assert XI == Operator.make_tensor1(2, Gate.X, 0)
    assert make_op("IIZII") == Operator.make_tensor1(5, Gate.Z, 2)

    SI = Operator.make_phase_op(2, [0], 1.j)
    assert SI != II
    assert SI != ZI
    assert SI*SI == ZI

    assert Operator.make_tensor1(2, Gate.S, 0) == SI

    spec = "ZXIZXIZIXIZ"
    A = make_op(spec)
    assert A*A == Operator.make_I(A.n)

    #ops = [getattr(Gate, c) for c in "ZXIZXIZIXIZ"]
    #B = Operator.make_tensor(ops)
    #assert B == A

    A = Operator.make_cz(2)
    B = Operator.make_control(2, Gate.Z, 0, 1)
    assert A == B

    n = 5
    A = Operator.make_cz(n)
    v = numpy.zeros((2,)*n, dtype=scalar)
    v[:] = 1
    u = A(v)

    for idx in numpy.ndindex(v.shape):
        if idx == (1,)*n:
            assert u[idx] == -1
        else:
            assert u[idx] == +1





"""
Construct transversal S gate on a folded surface code.
See: https://arxiv.org/abs/1603.02286
"""


class Lattice(object):
    def __init__(self, keys, d=2):
        items = set(keys)
        n = len(items)
        assert n <= 13, "too big!"
        assert n == len(keys)
        self.keys = keys
        self.lookup = {key:idx for (idx, key) in enumerate(keys)}
        self.items = items
        self.n = n
        self.d = d # qudit dimension

    def get_idxs(self, keys):
        lookup = self.lookup
        idxs = [lookup[key] for key in keys if lookup.get(key) is not None]
        return idxs

    def make_state(self, idxs=None):
        "computational basis state"
        n = self.n
        v = numpy.zeros(self.d**n, dtype=scalar)
        if idxs is not None:
            idx = sum([2**i for i in idxs])
            v[idx] = 1.
        return v

    @classmethod
    def make_surface(self, row0, row1, col0, col1):
        keys = set()
        zops = []
        xops = []
#        for i in range(row0, row1):
#          for j in range(col0, col1):
#            if row==0 and j%2==0
            


def main():

    eq = numpy.allclose

    keys = [
        (0,0,0), (0,1,0), (0,2,0), 
             (0,0,1), (0,1,1),
        (1,0,0), (1,1,0), (1,2,0), 
             (1,0,1), (1,1,1),
        (2,0,0), (2,1,0), (2,2,0), 
    ]
    assert len(keys) == 13
    lattice = Lattice(keys)
    n = lattice.n

    rows = 3
    cols = 3
    zops, xops = [], []
    for i in range(rows):
      for j in range(cols):
        keys = [(i, j, 0), (i, j, 1), (i-1, j, 1), (i, j+1, 0)]
        idxs = lattice.get_idxs(keys)
        if len(idxs)>=3:
            zops.append(idxs)
        #    A = lattice.make_op(Z, idxs)
        #    print(A.shape)

        keys = [(i, j, 0), (i, j, 1), (i+1, j, 0), (i, j-1, 1)]
        idxs = lattice.get_idxs(keys)
        if len(idxs)>=3:
            xops.append(idxs)

    v0 = lattice.make_state()
    v0[0] = 1

    idxs = lattice.get_idxs([(0,0,0), (0,1,0), (0,2,0)])
    v1 = lattice.make_state(idxs)

    get_idxs = lattice.get_idxs
    #make_op = lattice.make_op
    #make_control = lattice.make_control
    make_op = Operator.make_op

    stabs = []
    for opi in xops:
        B = Operator.make_xop(n, opi)
        stabs.append(B)
        v0 = v0 + B*v0
        v1 = v1 + B*v1

    #print(v0.shortstr())
    #print(v1.shortstr())

    #v[(1,)*lattice.n] = 1.
    for opi in zops:
        A = Operator.make_zop(n, opi)
        stabs.append(A)
        assert eq(A*v0, v0)
        v1 = v1 + A*v1

    for opi in zops:
        A = Operator.make_zop(n, opi)
        assert eq(A*v1, v1)

    if argv.slow:
        In = Operator.make_I(n)
        for A in stabs:
            assert A*A == In
            for B in stabs:
                assert A*B == B*A
                print("/", end="", flush=True)
        print()

    v0 /= numpy.linalg.norm(v0)
    v1 /= numpy.linalg.norm(v1)

    braket = lambda a, b : numpy.dot(a.conj(), b)
    r = braket(v0, v1)
    assert abs(r) < EPSILON, abs(r)
    #print("v0 =", v0.shortstr())
    #print("v1 =", v1.shortstr())

    geti = lambda idx : lattice.get_idxs([tuple(int(i) for i in idx)])[0]

    A =   Operator.make_control(n, Z, geti('100'), geti('010'))
    assert A == Operator.make_control(n, Z, geti('010'), geti('100'))
    A = A*Operator.make_control(n, Z, geti('200'), geti('020'))
    A = A*Operator.make_control(n, Z, geti('101'), geti('011'))
    A = A*Operator.make_control(n, Z, geti('210'), geti('120'))

    op = [I] * n
    op[geti("000")] = S
    op[geti("001")] = Sd
    op[geti("110")] = S
    op[geti("111")] = Sd
    op[geti("220")] = S
    A = A*Operator.make_tensor(op)

    # check we have a logical S gate
    assert eq(v0, A*v0)
    assert eq(1.j*v1, A*v1)

    print(".")
    if argv.slow:
        B = A*A*A*A
        assert B == Operator.make_I(n)

    for opi in xops:
        B = Operator.make_xop(n, opi)
        assert A*B == B*A

    for opi in zops:
        B = Operator.make_zop(n, opi)
        assert A*B == B*A
        print(".")



if __name__ == "__main__":

    if argv.test:
        test_op()
    else:
        main()

    print("OK")

