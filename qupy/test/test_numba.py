#!/usr/bin/env python3

"""
Construct transversal S gate on a folded surface code.
See: https://arxiv.org/abs/1603.02286

Use numba accelerated operators.

"""

import sys
import math
from random import shuffle, seed, choice
from functools import reduce
from operator import mul, matmul

import psutil
proc = psutil.Process()
def showmem(desc):
    print(desc)
    print(proc.memory_info())
def showmem(desc):
    pass

import numpy
import numba as nb

from qupy.argv import argv
is_real = False
if argv.complex64:
    from qupy import scalar
    scalar.scalar = numpy.complex64
    scalar.EPSILON = 1e-6
elif argv.float64:
    from qupy import scalar
    scalar.scalar = numpy.float64
    scalar.EPSILON = 1e-6
    is_real = True
elif argv.float32:
    from qupy import scalar
    scalar.scalar = numpy.float32
    scalar.EPSILON = 1e-6
    is_real = True

from qupy.dense import Qu, Gate, Vector, EPSILON, scalar
from qupy.dense import genidx, bits, is_close, on, off, scalar
from qupy.dense import commutator, anticommutator
from qupy.tool import cross
from qupy.util import mulclose

print("scalar:", scalar)

r2 = math.sqrt(2)

DEBUG = argv.debug

if is_real:
    I, X, Z = Gate.I, Gate.X, Gate.Z
    
    assert X*X == I
    assert Z*Z == I
    S, T = None, None

else:
    I, X, Z, S, T = Gate.I, Gate.X, Gate.Z, Gate.S, Gate.T
    
    assert X*X == I
    assert Z*Z == I
    assert S*S == Z
    #assert (T*T).is_close(S, 1e-6)
    assert T*T == S
    
    Sd = S.dag()
    assert S*Sd == I


class Operator(object):
    def __init__(self, n, d=2, inverse=None, self_inverse=False):
        self.n = n
        self.N = d**n
        self.d = d
        self.shape = (d,)*n
        self.dtype = scalar
        if self_inverse:
            inverse = self
        self.inverse = inverse

#    @property
#    def inverse(self):

    def __str__(self):
        return "%s(n=%d)"%(self.__class__.__name__, self.n)

    def __call__(self, v, u=None):
        try:
            u = v.copy()
            return u
        except:
            print("%s.__call__(%s, %s)"%(self, v, u))
            raise

    DEBUG_EQ = False
    def __eq_slow__(lhs, rhs):
        assert lhs.n == rhs.n
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
            if Operator.DEBUG_EQ:
                print(".", end="", flush=True)
        if Operator.DEBUG_EQ:
            print()
        return True

    def __eq_fast_big__(lhs, rhs):
        #return numpy.allclose(lhs.todense(), rhs.todense())
        n = lhs.n
        N = 2**n
        u = numpy.random.normal(size=(N,))
        showmem("__eq_fast_big__: random")
        u = u.astype(scalar)
        showmem("__eq_fast_big__: lhs")
        lhs = lhs(u)
        showmem("__eq_fast_big__: rhs")
        rhs = rhs(u)
        showmem("__eq_fast_big__: allclose")
        M = 2**24
        idx = 0
        close = True
        while idx < N and close:
            close = close and numpy.allclose(lhs[idx:idx+M], rhs[idx:idx+M])
            idx += M
        return close

    def __eq_fast__(lhs, rhs):
        assert lhs.n == rhs.n
        if lhs.n > 24:
            return lhs.__eq_fast_big__(rhs) # <--- return
        n = lhs.n
        N = 2**n
        u = numpy.random.normal(size=(N,))
        u = u.astype(scalar)
        lhs = lhs(u)
        rhs = rhs(u)
        if not numpy.allclose(lhs, rhs):
            return False
        return True

    #__eq__ = __eq_slow__
    __eq__ = __eq_fast__

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

    def trace(self):
        N = 2**self.n
        #if self.n <= 8:
        #    A = self.todense()
        #    A.shape = (N, N)
        #    r = numpy.trace(A)
        #    return r
        u = numpy.zeros(N, dtype=scalar)
        r = 0j
        for i in range(N):
            u[i] = 1
            v = self(u)
            r += v[i]
            u[i] = 0
        if abs(r.imag)<EPSILON:
            r = r.real
        return r

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
        I = Operator(n, self_inverse=True)
        return I

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
        if not idxs:
            return cls.make_I(n)
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
        return Numop(n, func, self_inverse=True)

    @classmethod    
    def make_ccz(cls, n, i, j, k):
        if DEBUG:print("make_ccz", n, i, j, k)
        stmt = """
        def func(v, u):
            sign = 1.
            for src in range(%d):
                if (src & %d) and (src & %d) and (src & %d):
                    sign = -1
                u[src] = sign*v[src]
        """ % (2**n, 2**i, 2**j, 2**k)
        #print(stmt)
        func = mkfunc(stmt)
        return Numop(n, func, self_inverse=True)

    @classmethod    
    def make_phase_op(cls, n, idxs, phase, inverse=None):
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
        op = Numop(n, func, inverse=inverse)
        if inverse is None:
            assert abs(phase)>EPSILON
            op.inverse = cls.make_phase_op(n, idxs, 1./phase, op)
        return op

    @classmethod    
    def make_xop(cls, n, idxs):
        if DEBUG:print("make_xop", n, idxs)
        if not idxs:
            return cls.make_I(n)
        stmt = """
        def func(v, u):
          for src in range(%d):
            tgt = src
            for i in %s:
                tgt ^= i
            u[tgt] = v[src]
        """ % (2**n, [2**i for i in idxs])

        func = mkfunc(stmt)
        return Numop(n, func, self_inverse=True)

    @classmethod
    def make_op(cls, spec):
        n = len(spec)
        zidxs = [i for i in range(n) if spec[i]=='Z']
        xidxs = [i for i in range(n) if spec[i]=='X']
        assert 'Y' not in spec
        op = Operator(n, self_inverse=True)
        if zidxs:
            op = Operator.make_zop(n, zidxs)
            assert op.inverse is op
        if xidxs:
            xop = Operator.make_xop(n, xidxs)
            assert xop.inverse is xop
            op = op * xop
#            assert op.inverse is not None, type(op) # FIX FIX FIX
        return op

    @classmethod    
    def make_cnz(cls, n):
        if DEBUG:print("make_cnz", n)
        stmt = """
        def func(v, u):
            u[:] = v[:]
            u[%d] = -v[%d]
        """ % (2**n-1, 2**n-1)
        func = mkfunc(stmt)
        return Numop(n, func, self_inverse=True)

#    @classmethod
#    def XXX_make_tensor(cls, ops):
#        assert 0, "FAIL"
#        n = len(ops)
#        N = 2**n
#        d = 2
#        code = Source()
#        code.append("def func(v, u):").indent()
#        code.append("for j in range(%d):"%N).indent()
#        bit = 1
#        for op in ops:
#            assert op.shape == (2, 2)
#            for i in range(N):
#                code.append("if j & %d == 0:"%bit).indent()
#                code.dedent()
#                code.append("else:").indent()
#                code.dedent()
#                code.append("u[%d] = r"%i)
#            bit *= 2
#    
#        #code.append("u[j] = v[j]").indent()
#        
#        print("make_tensor:")
#        print('\n'.join(code.lines))
#        func = code.mkfunc()
#        return Numop(n, func)

    @classmethod
    def make_tensor1(cls, n, A, idx, inverse=None):
        N = 2**n
        d = 2
        assert A.shape == (2,2)
        code = Source()
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
        
        #print("make_tensor1:")
        #print('\n'.join(code.lines))
        func = code.mkfunc()
        op = Numop(n, func, inverse=inverse)
        if inverse is None:
            Ai = A.inverse()
            op.inverse = cls.make_tensor1(n, Ai, idx, inverse=op)
        return op

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
    def make_control(cls, n, A, tgt, src, inverse=None):
        N = 2**n
        d = 2
        assert A.shape == (2,2)
        code = Source()
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
        
        #print("make_control:")
        #print('\n'.join(code.lines))
        func = code.mkfunc()
        op = Numop(n, func, inverse=inverse)
        if inverse is None:
            Ai = A.inverse()
            op.inverse = cls.make_control(n, Ai, tgt, src, inverse=op)
        return op

    @classmethod
    def make_swap(cls, n, tgt, src):
        N = 2**n
        d = 2
        code = Source()
        code.append("def func(v, u):").indent()
        code.append("for j in range(%d):"%N).indent()
        code.append("i = j")
        code.append("if (j&%d==0) != (i&%d==0):"%(2**src, 2**tgt)).indent()
        code.append("i ^= %d # flip bit"%(2**tgt)).dedent()
        code.append("if (j&%d==0) != (i&%d==0):"%(2**tgt, 2**src)).indent()
        code.append("i ^= %d # flip bit"%(2**src)).dedent()
        code.append("u[i] = v[j]")
        code.dedent()
        #print("make_swap:")
        #print('\n'.join(code.lines))
        #print()
        func = code.mkfunc()
        return Numop(n, func, self_inverse=True)



class Source(object):
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
    def __init__(self, n, func, d=2, **kw):
        Operator.__init__(self, n, d, **kw)
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
        assert u.shape == (self.N,), "%s != (%s,)"%(u.shape, self.N)
        if v is None:
            v = numpy.zeros(shape, dtype=self.dtype)
        self.func(u, v)
        if reshape:
            v.shape = self.shape
        return v


class BinOp(Operator):
    def __init__(self, lhs, rhs, inverse=None):
        assert lhs.n == rhs.n
        Operator.__init__(self, lhs.n, inverse=inverse)
        self.lhs = lhs
        self.rhs = rhs

class AssocOp(Operator):
    def __init__(self, *_items, **kw):
        items = []
        for item in _items:
            if isinstance(item, self.__class__):
                items += item.items
            else:
                items.append(item)
        Operator.__init__(self, items[0].n, **kw)
        self.items = items


class AddOp(AssocOp):
    def __call__(self, u, v=None):
        shape = (2**self.n,)
        if v is None:
            v = numpy.zeros(shape, dtype=self.dtype)
        for item in self.items:
            v += item(u)
        return v


class SubOp(BinOp):
    def __call__(self, u):
        lhs = self.lhs(u)
        rhs = self.rhs(u)
        return lhs - rhs


class MulOp(AssocOp):
    def __init__(self, *items, **kw):
        AssocOp.__init__(self, *items, **kw)
        if self.inverse is not None:
            return
        inverse = [op.inverse for op in reversed(self.items) if op.inverse is not None]
        if len(inverse) < len(items):
            return
        self.inverse = MulOp(*inverse, inverse=self)

    def __call__(self, u):
        v = u
        for item in reversed(self.items):
            v = item(v)
        return v



class RMulOp(Operator):
    def __init__(self, lhs, r, inverse=None):
        if inverse is None and abs(r)>EPSILON and lhs.inverse is not None:
            inverse = RMulOp(lhs.inverse, 1./r, self)
        Operator.__init__(self, lhs.n, inverse=inverse)
        self.lhs = lhs
        self.r = r

    def __call__(self, u):
        #print("RMulOp.__call__", u)
        u = self.lhs(u)
        u = self.r * u
        #print("\t\t", u)
        return u



class Space(object):

    cache = {}
    def __init__(self, n):
        self.n = n
        self.I = Operator.make_I(n)

    def make_tensor1(self, A, idx, **kw):
        n = self.n
        assert 0<=idx<n
        op = Operator.make_tensor1(n, A, idx, **kw)
        return op

    def make_xop(self, idxs):
        n = self.n
        key = "make_xop", n, tuple(idxs)
        if key in self.cache:
            #print("+", end="", flush=True)
            return self.cache[key]
        for idx in idxs:
            assert 0<=idx<n
        op = Operator.make_xop(n, idxs)
        self.cache[key] = op
        return op

    def make_zop(self, idxs):
        n = self.n
        key = "make_zop", n, tuple(idxs)
        if key in self.cache:
            #print("+", end="", flush=True)
            return self.cache[key]
        for idx in idxs:
            assert 0<=idx<n
        op = Operator.make_zop(n, idxs)
        self.cache[key] = op
        return op

    def make_op(self, decl):
        n = self.n
        assert len(decl) == n
        #print("make_op", decl)
        zidxs = [i for i in range(n) if decl[i] in 'ZY']
        xidxs = [i for i in range(n) if decl[i] in 'XY']
        #print("make_op", decl, zidxs, xidxs)
        zop = self.make_zop(zidxs)
        xop = self.make_xop(xidxs)
        lhs = zop * xop
        if decl.count('Y') % 2:
            lhs = 1j * lhs
            #print(lhs, lhs.r)
            #print(lhs.inverse, lhs.inverse.r)
        return lhs

    def make_control(self, A, i, j):
        n = self.n
        assert 0<=i<n
        assert 0<=j<n
        assert i!=j
        op = Operator.make_control(n, A, i, j)
        return op

    def make_cz(self, i, j):
        return self.make_control(Z, i, j)

    def make_ccz(self, i, j, k):
        n = self.n
        assert 0<=i<n
        assert 0<=j<n
        assert 0<=k<n
        assert i!=j!=k
        assert i!=k
        op = Operator.make_ccz(n, i, j, k)
        return op
    
    def get_basis(self):
        count = 0
        for decl in cross(["IXZY"]*self.n):
            decl = ''.join(decl)
            op = self.make_op(decl)
            yield decl, op
            count += 1
            #print(".", end="", flush=True)
        #print()
        assert count==4**self.n

    def opstr(self, P):
        items = []
        N = 2**self.n
        for k,v in self.get_basis():
            #assert v*v == self.I
            #assert v.inverse == v # yes...
            r = (v.inverse*P).trace() / N 
            if abs(r)<EPSILON:
                #print("0", end="",)
                continue
            if abs(r.real - r)<EPSILON:
                r = r.real
                if abs(int(round(r)) - r)<EPSILON:
                    r = int(round(r))
            if r==1:
                items.append("%s"%(k,))
            elif r==-1:
                items.append("-%s"%(k,))
            else:
                items.append("%s*%s"%(r, k))
            #print("[%s]"%k, end="", flush=True)
        s = "+".join(items) or "0"
        s = s.replace("+-", "-")
        return s



def test():

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
        assert op.inverse is not None, str(op)
        assert op*op.inverse == II
    assert ZI*XI != XI*ZI
    assert ZI*XI == -(XI*ZI) 

    assert II == Operator.make_tensor1(2, Gate.I, 1)
    assert XI == Operator.make_tensor1(2, Gate.X, 0)
    assert XI == Operator.make_tensor1(2, Gate.X, 0).inverse
    assert make_op("IIZII") == Operator.make_tensor1(5, Gate.Z, 2)

    SI = Operator.make_phase_op(2, [0], 1.j)
    assert SI != II
    assert SI != ZI
    assert SI*SI == ZI
    assert SI * SI.inverse == II

    assert Operator.make_tensor1(2, Gate.S, 0) == SI

    spec = "ZXIZXIZIXIZ"
    A = make_op(spec)
    assert A*A == Operator.make_I(A.n)

    #ops = [getattr(Gate, c) for c in "ZXIZXIZIXIZ"]
    #B = Operator.make_tensor(ops)
    #assert B == A

    A = Operator.make_cnz(2)
    B = Operator.make_control(2, Gate.Z, 0, 1)
    assert A == B

    n = 5
    A = Operator.make_cnz(n)
    v = numpy.zeros((2,)*n, dtype=scalar)
    v[:] = 1
    u = A(v)

    for idx in numpy.ndindex(v.shape):
        if idx == (1,)*n:
            assert u[idx] == -1
        else:
            assert u[idx] == +1


    n, src, tgt = 5, 1, 2
    CN1 = Operator.make_control(n, Gate.X, src, tgt)
    CN2 = Operator.make_control(n, Gate.X, tgt, src)
    swap = Operator.make_swap(n, src, tgt)
    assert swap == CN1*CN2*CN1
    assert swap != Operator.make_I(n)
    assert swap*swap == Operator.make_I(n)

    CCZ = Operator.make_ccz(3, 0, 1, 2)
    CCZ = CCZ.todense().reshape((8, 8))
    lhs = numpy.identity(8)
    lhs[7, 7] = -1
    assert numpy.allclose(CCZ, lhs)

    n = 1
    space = Space(n)

    for k, op in space.get_basis():
        assert op * op.inverse == space.I
        assert space.opstr(op) == k

    n = 3
    space = Space(n)

    for k, op in space.get_basis():
        assert op * op.inverse == space.I
        assert space.opstr(op) == k



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
        "computational basis _state"
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
            

eq = numpy.allclose


def main_13():

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

    if argv.dual:
        zops, xops = xops, zops

    v0 = lattice.make_state()
    v0[0] = 1

    if argv.dual:
        idxs = lattice.get_idxs([(0,0,0), (1,0,0), (2,0,0)])
    else:
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

    In = Operator.make_I(n)
    for A in stabs:
        assert A*A == In
        for B in stabs:
            assert A*B == B*A

    v0 /= numpy.linalg.norm(v0)
    v1 /= numpy.linalg.norm(v1)

    braket = lambda a, b : numpy.dot(a.conj(), b)
    r = braket(v0, v1)
    assert abs(r) < EPSILON, abs(r)
    #print("v0 =", v0.shortstr())
    #print("v1 =", v1.shortstr())

    geti = lambda desc : lattice.get_idxs([tuple(int(i) for i in desc)])[0]

    H = Operator.make_I(n)
    for desc in "000 001 110 111 220".split():
        H *= Operator.make_tensor1(n, Gate.H, geti(desc))

    def make_h2(idx, jdx):
        HH =  Operator.make_swap(n, idx, jdx)
        HH *= Operator.make_tensor1(n, Gate.H, idx)
        HH *= Operator.make_tensor1(n, Gate.H, jdx)
        return HH
    swap =  make_h2(geti('010'), geti('100'))
    swap *= make_h2(geti('101'), geti('011'))
    swap *= make_h2(geti('200'), geti('020'))
    swap *= make_h2(geti('210'), geti('120'))
    H *= swap

    #H = Operator.make_I(n)
    #for i in range(n):
    #    H *= Operator.make_tensor1(n, Gate.H, i)
    #assert H*H == Operator.make_I(n)

    if argv.H:
        print("H:")
        for a in stabs:
            b = H*a*H
            for c in stabs:
                print(int(b==c), end=" ", flush=True)
            print()

    assert eq((1./r2)*(v0+v1), H(v0))
    assert eq((1./r2)*(v0-v1), H(v1))

    assert H*H == Operator.make_I(n)

    #return

    if 0:
        P = None
        for ops in cross([(None, op) for op in stabs]):
            ops = [op for op in ops if op is not None] or [Operator.make_I(n)]
            op = reduce(mul, ops)
            P = op if P is None else op+P
        print(len(P.items))
    else:
        P = None
        In = Operator.make_I(n)
        for op in stabs:
            A = (In + op)
            P = A if P is None else A*P

    assert P*P == (2**len(stabs))*P

    Sgate =   Operator.make_control(n, Z, geti('100'), geti('010'))
    assert Sgate == Operator.make_control(n, Z, geti('010'), geti('100'))
    Sgate = Sgate*Operator.make_control(n, Z, geti('200'), geti('020'))
    Sgate = Sgate*Operator.make_control(n, Z, geti('101'), geti('011'))
    Sgate = Sgate*Operator.make_control(n, Z, geti('210'), geti('120'))

    op = [I] * n
    op[geti("000")] = S
    op[geti("001")] = Sd
    op[geti("110")] = S
    op[geti("111")] = Sd
    op[geti("220")] = S
    Sgate = Sgate*Operator.make_tensor(op)

    # check we have a logical S gate
    assert eq(v0, Sgate*v0)
    assert eq(1.j*v1, Sgate*v1)

    assert(Sgate*P == P*Sgate)

    B = Sgate*Sgate*Sgate*Sgate
    assert B == Operator.make_I(n)


def main_vasmer():

    make_op = Operator.make_op
    make_I = Operator.make_I

    n = 10
    I = make_I(n)
    stabs = ("XXIXXIIXII IXXXIXIIXI IIIXXXXIIX " 
        "ZZIIIIIIZI IIIZZIIIZI IZIZIIIIIZ IIZIIZIIIZ IIIZIZIZII IIIIZIZZII")
    stabs = [make_op(decl) for decl in stabs.split()]

    for a in stabs:
        for b in stabs:
            assert a*b == b*a
        #print("/", end="", flush=True)

    P = None
    for ops in cross([(None, op) for op in stabs]):
        ops = [op for op in ops if op is not None] or [Operator.make_I(n)]
        op = reduce(mul, ops)
        P = op if P is None else op+P

    assert P*P == (2**len(stabs))*P

    T  = Operator.make_tensor1(n,  Gate.T, 0)
    T *= Operator.make_tensor1(n, ~Gate.T, 1)
    T *= Operator.make_tensor1(n,  Gate.T, 2)
    T *= Operator.make_tensor1(n,  Gate.T, 3)
    T *= Operator.make_tensor1(n, ~Gate.T, 4)
    T *= Operator.make_tensor1(n, ~Gate.T, 5)
    T *= Operator.make_tensor1(n,  Gate.T, 6)

    CCZ = Operator.make_ccz(n, 7, 8, 9)
    T *= CCZ

    assert T*P == P*T


from qupy.ldpc.solve import (
        identity2, kron, dot2, rank, int_scalar, 
        parse, remove_dependent, zeros2, rand2, shortstr, all_codes,
        find_kernel )


def all_nonzero_codes(m, n):
    for H in all_codes(m, n):
        rows = H.sum(0)
        if 0 in rows:
            continue
        yield H


class Code(object):
    def __init__(self, Hs, **kw):
        self.__dict__.update(kw)
        if type(Hs) is str:
            Hs = Hs.split()
        n = None
        for H in Hs:
            assert n is None or len(H)==n
            n = len(H)
            assert H.count("I") + H.count("Z") + H.count("X") == n, "%s wtf?"%H
        m = len(Hs)
        space = Space(n)

        stabs = []
        for i in range(m):
            idxs = [j for j in range(n) if Hs[i][j]=='Z']
            zop = space.make_zop(idxs)
            idxs = [j for j in range(n) if Hs[i][j]=='X']
            xop = space.make_xop(idxs)
            op = zop*xop
            stabs.append(op)

        # code projector:
        P = None
        for op in stabs:
            A = (space.I + op)
            P = A if P is None else A*P
        P = 1./(2**len(stabs)) * P

        self.Hs = Hs
        self.stabs = stabs
        self.space = space
        self.P = P
        self.m = m
        self.n = n

    def __getattr__(self, name):
        meth = getattr(self.space, name, None)
        if meth is None:
            raise AttributeError("Code object has no attribute %r"%name)
        return meth

    def __str__(self):
        return "Code(n=%d, m=%d)"%(self.n, self.m)

    def check(self):
        stabs = self.stabs
        for a in stabs:
          for b in stabs:
            assert a*b==b*a
        P = self.P
        #assert P*P == (2**len(stabs))*P
        assert P*P == P


def test_weight_enums():

    # https://arxiv.org/pdf/quant-ph/9610040.pdf

    code = argv.get("code")

    if code == "gcolor":
        n = 15
        ops = """
        1111...........
        ....1111.......
        11..11.........
        ..11..11.......
        1.1.1.1........
        .1.1.1.1.......
        ........1111...
        ........11..11.
        ........1.1.1.1
        11......11.....
        ..11......11...
        1.1.....1.1....
        .1.1.....1.1...
        ....11......11.
        ....1.1.....1.1
        1...1...1...1..
        .1...1...1...1.
        ..1...1...1...1
        """.strip().split()
        assert len(ops) == 18

        space = Space(n)
        xops = [space.make_xop([i for i in range(n) if op[i]=='1']) for op in ops]
        zops = [space.make_zop([i for i in range(n) if op[i]=='1']) for op in ops]
        #print(space.opstr(xops[0])) # too big...
        #return
        P = xops[0] 
        for op in xops[1:] + zops:
            P = P + op

        # probably hopeless to try this...
        return

    else:
        if code == "five":
            Hs = "XZZXI IXZZX XIXZZ ZXIXZ" # five qubit code
        elif code == "steane":
            Hs = "XXXXIII XXIIXXI XIXIXIX ZZZZIII ZZIIZZI ZIZIZIZ" # steane code
        else:
            assert 0, code
        code = Code(Hs)
        code.check()
        print(code)
        P = code.P
        n = code.n
        space = code.space
        

    weights = dict((w, []) for w in range(n+1))
    #print(weights)
    get_weight = lambda decl : len(decl)-decl.count("I")
    for decl,op in space.get_basis():
        weights[get_weight(decl)].append(op)

    Ptr = P.trace()

    U = None

    if 0:
        U = Gate.random_unitary(2)
        #U = U.v
        #print(U)
        #Ui = U.transpose().conj()
        #print(numpy.dot(U, Ui))
        idx = 0
        U = Operator.make_tensor1(n, U, idx)
        #U = space.make_tensor1(U, idx, inverse=~U)

    elif 0:

        Us = [Gate.random_unitary(2) for idx in range(n)]
        U = Operator.make_tensor(Us)


    def write(x):
        if abs(x)<EPSILON:
            x = 0.
        if abs(x - round(x)) < EPSILON:
            x = int(round(x))
        print(x, end=" ", flush=True)

    if U is not None:
        P = U * P * U.inverse

    for d in range(n+1):
        Ad = 0.
        for op in weights[d]:
            val = (op*P).trace() * (op.inverse * P).trace()
            Ad += val / (Ptr**2)
        write(Ad)
    print()
    
    for d in range(n+1):
        Bd = 0.
        for op in weights[d]:
            val = (op*P*op.inverse * P).trace()
            Bd += val / Ptr
        write(Bd)
    print()
    


class CSSCode(object):
    def __init__(self, Hz, Hx, Lz=None, Lx=None, **kw):
        self.__dict__.update(kw)
        mz, n = Hz.shape
        mx, nx = Hx.shape
        assert n==nx
        assert rank(Hz)==mz
        assert rank(Hx)==mx
        xstabs = []
        zstabs = []
        space = Space(n)

        for i in range(mz):
            idxs = [j for j in range(n) if Hz[i, j]]
            op = space.make_zop(idxs)
            zstabs.append(op)

        for i in range(mx):
            idxs = [j for j in range(n) if Hx[i, j]]
            op = space.make_xop(idxs)
            xstabs.append(op)

        # code projector:
        P = None
        for op in zstabs + xstabs:
            A = (space.I + op)
            P = A if P is None else A*P

        if Lz is not None:
            assert Lx is not None
            assert len(Lz)==len(Lx)
            k = len(Lz)
            assert k==n-mx-mz
            assert dot2(Lz, Hx.transpose()).sum() == 0
            assert dot2(Lx, Hz.transpose()).sum() == 0
            #print(shortstr(dot2(Lz, Lx.transpose())))

            zlogops = []
            xlogops = []

            for i in range(k):
                idxs = [j for j in range(n) if Lz[i, j]]
                op = space.make_zop(idxs)
                zlogops.append(op)
                
                idxs = [j for j in range(n) if Lx[i, j]]
                op = space.make_xop(idxs)
                xlogops.append(op)
    
        else:
            zlogops = None
            xlogops = None

        self.Hz = Hz
        self.Hx = Hx
        self.zstabs = zstabs
        self.xstabs = xstabs
        self.stabs = zstabs + xstabs
        self.zlogops = zlogops
        self.xlogops = xlogops
        self.space = space
        self.mz = mz
        self.mx = mx
        self.n = n
        self.k = n-mx-mz
        assert self.k >= 0
        self.P = P
        #self.check()

    def __str__(self):
        return "CSSCode(n=%d, mx=%d, mz=%d, k=%d)"%(self.n, self.mx, self.mz, self.k)

    def check(self):
        stabs = self.stabs
        for a in stabs:
          for b in stabs:
            assert a*b==b*a
        P = self.P
        assert P*P == (2**len(stabs))*P
        xlogops, zlogops = self.xlogops, self.zlogops
        if xlogops is not None:
            for i, a in enumerate(stabs):
                for j, op in enumerate(zlogops):
                    assert a*op == op*a, (i, j)
                for j, op in enumerate(xlogops):
                    assert a*op == op*a, (i, j)
                    #print( a*op == op*a, (i, j) , end=" ")
                #print()
        #print("CSSCode.check(): OK")

    def __add__(self, other):
        Hz0 = numpy.concatenate((self.Hz, zeros2(self.mz, other.n)), axis=1)
        Hx0 = numpy.concatenate((self.Hx, zeros2(self.mx, other.n)), axis=1)
        Hz1 = numpy.concatenate((zeros2(other.mz, self.n), other.Hz), axis=1)
        Hx1 = numpy.concatenate((zeros2(other.mx, self.n), other.Hx), axis=1)
        Hz = numpy.concatenate((Hz0, Hz1))
        Hx = numpy.concatenate((Hx0, Hx1))
        return CSSCode(Hz, Hx)

    def longstr(self):
        s = '\n'.join([
            str(self), "Hz:", shortstr(self.Hz), "Hx:", shortstr(self.Hx) ])
        return s

    def get_encoded(self, idx=0):
        k = self.k
        n = self.n
        assert 0<=idx<2**k
        v = numpy.zeros((2**n,), dtype=scalar)
        v[0] = 1
        v = self.P*v
        for i in range(k):
            #print(i, len(self.xlogops))
            if idx & (2**i):
                v = self.xlogops[i] * v
        r = numpy.linalg.norm(v)
        assert r>EPSILON
        v /= r
        return v

    def __getattr__(self, name):
        meth = getattr(self.space, name, None)
        if meth is None:
            raise AttributeError("CSSCode object has no attribute %r"%name)
        return meth


def hypergraph_product(A, B):
    #print("hypergraph_product: A=%s, B=%s"%(A.shape, B.shape))

    ma, na = A.shape
    mb, nb = B.shape

    Ima = identity2(ma)
    Imb = identity2(mb)
    Ina = identity2(na)
    Inb = identity2(nb)

    Hz0 = kron(Ina, B.transpose()), kron(A.transpose(), Inb)
    Hz = numpy.concatenate(Hz0, axis=1) # horizontal concatenate

    Hx0 = kron(A, Imb), kron(Ima, B)
    #print("Hx0:", Hx0[0].shape, Hx0[1].shape)
    Hx = numpy.concatenate(Hx0, axis=1) # horizontal concatenate

    assert dot2(Hx, Hz.transpose()).sum() == 0

    n = Hz.shape[1]
    assert Hx.shape[1] == n

    Hzi = remove_dependent(Hz)
    Hxi = remove_dependent(Hx)

    code = CSSCode(Hzi, Hxi)
    return code


def get_swap(n):
    swap = {}
    for i in range(n):
      for j in range(n):
        k0 = i + j*n
        k1 = j + i*n
        swap[k0] = k1
    return swap

def schur(H):
    m, n = H.shape
    Ht = H.transpose()

    In = identity2(n)
    Im = identity2(m)
    Hz0 = kron(In, H), kron(Ht, Im)
    Hz = numpy.concatenate(Hz0, axis=1) # horizontal concatenate

    Hx0 = kron(H, In), kron(Im, Ht)
    #print("Hx0:", Hx0[0].shape, Hx0[1].shape)
    Hx = numpy.concatenate(Hx0, axis=1) # horizontal concatenate

    assert dot2(Hx, Hz.transpose()).sum() == 0
    assert Hx.shape[1] == Hz.shape[1]

    swap = get_swap(n)
    for (i,j) in get_swap(m).items():
        swap[i+n*n] = j+n*n
    vfix = set()
    hfix = set()
    lx = zeros2(n*n + m*m)
    for i in range(n):
        k = i + i*n
        lx[k] = 1
        assert swap[k] == k
        vfix.add(k)
    for i in range(m):
        k = n*n + i + i*m
        lx[k] = 1
        assert swap[k] == k
        hfix.add(k)
    lx.shape = n*n+m*m, 1
    #print(dot2(Hz, lx))
    assert dot2(Hz, lx).sum() == 0
    #print("lx:")
    #print(lx.transpose())

    #return

    Hzi = remove_dependent(Hz)
    Hxi = remove_dependent(Hx)

    code = CSSCode(Hzi, Hxi)
    print(code)

    A = None
    for i in range(n*n + m*m):
        j = swap[i]
        #if n*n+m*m-1 in [i,j]:
        #    break
        if j < i:
            continue
        if i==j:
            if i in vfix:
                B = code.make_tensor1(Gate.S, i)
            else:
                assert i in hfix
                B = code.make_tensor1(~Gate.S, i)
        else:
            B = code.make_control(Z, i, j)
        A = B if A is None else B*A

    assert(A*code.P == code.P*A)

    return code



def main_product():
    "homological product codes"

    m = argv.get("m", 2)
    n = argv.get("n", 3)
    assert m and n

    while 1:

        if argv.surface:
            H = parse("11. .11")
        elif argv.toric:
            H = parse("11. .11 1.1")
        elif argv.rand:
            while 1:
                H = rand2(m, n)
                if H.sum():
                    break
        else:
            H = parse("111 111")
        print("H:")
        print(shortstr(H))
    
        #code1 = hypergraph_product(H, H.transpose())
        #print(code1)
    
        code = schur(H)
        #assert code1.P == code.P
    
        if argv.check:
            code.check()

        if not argv.forever:
            break


def main_8T():
    """
    _Transversal T gate on the [[8,3,2]] colour code.
    https://earltcampbell.com/2016/09/26/the-smallest-interesting-colour-code
    https://arxiv.org/abs/1706.02717
    """

    Hz = parse("1111.... 11..11.. 1.1.1.1. 11111111")
    Hx = parse("11111111")
    Lz = parse("1...1... 1.1..... 11......")
    Lx = parse("1111.... 11..11.. 1.1.1.1.")

    code = CSSCode(Hz, Hx, Lz, Lx)
    code.check()

    T, Td = Gate.T, ~Gate.T
    A = None
    for i, B in enumerate([T, Td, Td, T, Td, T, T, Td]):
        B = code.make_tensor1(B, i)
        A = B if A is None else B*A

    assert A*code.P == code.P*A

    basis = []
    for idx in range(2**3):
        v = code.get_encoded(idx)
        basis.append(v)

    P = (1./2**len(code.stabs))*code.P
    assert P*P == P
    for u in basis:
        assert eq(P*u, u)

#    for u in basis:
#        for v in basis:
#            r = numpy.dot(v.conj(), u)
#            if abs(r.imag)<EPSILON:
#                r = r.real
#            print("%.2f"%r, end=" ")
#        print()
#    print()

    # encoded CCZ
    for i, u in enumerate(basis):
        u = A*u
        for j, v in enumerate(basis):
            r = numpy.dot(v.conj(), u)
            if i==j < 7:
                assert abs(r-1.) < EPSILON
            elif i==j==7:
                assert abs(r+1.) < EPSILON
            else:
                assert abs(r) < EPSILON

    if 0:
        n = code.n
        code = code + code + code
        print(code)
            
        A = None
        for i in range(n):
            B = code.make_ccz(i, i+n, i+2*n)
            A = B if A is None else B*A
    
        print( A*code.P == code.P*A ) # False ...


def main_mobius():
    """
    """

    Hx = """
    1  1  .  1  .  1  .  .  .   
    .  1  1  .  .  .  1  .  1     
    .  .  .  .  1  1  1  1  .    
    """.replace(" ", "")
    Hz = """
    1  .  .  1  .  .  .  .  . 
    .  1  .  1  1  .  1  .  . 
    1  .  1  .  .  1  1  .  .
    .  .  .  .  1  .  .  1  .
    .  1  .  .  .  1  .  1  1
    """.replace(" ", "")

    Hz = parse(Hz)
    Hx = parse(Hx)

    code = CSSCode(Hz, Hx)
    print(code)
    code.check()

    P = (1./2**len(code.stabs))*code.P
    assert P*P == P

    n = code.n
    gates = [Gate.S, ~Gate.S]
    opis = [(0, 1)]*n
    count = 0
    for idxs in cross(opis):
        A = None
        for i in range(n):
            B = code.make_tensor1(gates[idxs[i]], i)
            A = B if A is None else B*A
        Ad = None
        for i in range(n):
            B = code.make_tensor1(gates[1-idxs[i]], i)
            Ad = B if Ad is None else B*Ad
        assert A*Ad == code.I

        print(int(code.P*A == A*code.P) or '.', end="", flush=True)
        #print(code.opstr( A*code.P*Ad ))
    print()
    

def weakly_self_dual_codes(m, n, minfixed=2, trials=100):
    for Hz in all_nonzero_codes(m, n):
        for trial in range(trials):
            while 1:
                perm = list(range(n))
                shuffle(perm)
                fixed = [i for (i,j) in enumerate(perm) if i==j]
                if len(fixed)>=minfixed:
                    break
            Hx = Hz[:, perm]

            if dot2(Hz, Hx.transpose()).sum() == 0:
                #print("Hx:")
                #print(shortstr(Hz))
                #print("Hz:")
                #print(shortstr(Hx))
                #print("found")
                yield (Hz, Hx, perm)
                break


def main_self_dual():
    m = argv.get("m", 2)
    n = argv.get("n", 4)

    for Hz, Hx, perm in weakly_self_dual_codes(m, n):
    
        code = CSSCode(Hz, Hx)
        print(code)
    
        print("Hx:")
        print(shortstr(Hz))
        print("Hz:")
        print(shortstr(Hx))
        print(perm)
    
        #code.check()
    
        n = code.n
        A = None
        fixed = [i for i, j in enumerate(perm) if i==j]
        print("fixed:", fixed)
        for ss in cross([(Gate.S, ~Gate.S)]*len(fixed)):
            assert len(ss) == len(fixed)
            for i, j in enumerate(perm):
                #print("%d -> %d" % (i, j))
                if j<i:
                    continue
                if i==j:
                    op = ss[fixed.index(i)]
                    #print(op.shortstr())
                    B = code.make_tensor1(op, i)
                else:
                    B = code.make_control(Z, i, j)
                A = B if A is None else B*A
            print(".", end="", flush=True)
            if A*code.P == code.P*A:
                print("found", end="")
                break
        else:
            assert 0
        print()
    

def main_pair():
    m = argv.get("m", 3)
    n = argv.get("n", 7)

    from bruhat.action import Perm, Group

    for Hz, Hx, perm in weakly_self_dual_codes(m, n):

        g = Perm({i:j for (i,j) in enumerate(perm)}, list(range(n)))
        r = g.order()
        print(".", end="", flush=True)
        if r not in [2, 4, 8]:
            continue
        #if r != 8:
        #    continue
        print()
        print("order:", r)
        #if r==4:
        #    print("g --> g*g")
        #    g = g*g
        #    assert g.order() == 2
        #    perm = [g[i] for i in range(n)]
    
        code = CSSCode(Hz, Hx)
        print(code)
        print(code.longstr())
        #print(perm)
        code2 = code + code

        perm2 = list(perm)
        for (i, j) in enumerate(perm):
            perm2.append(n+j)
        #print(code2.longstr())
        #print(perm2)

        Hx = code2.Hz[:, perm2]
        assert eq(Hx, code2.Hx)

        A = None
        for i, j in enumerate(perm):
            B = code2.make_control(Z, i, n+j)
            A = B if A is None else B*A
        if A*code2.P == code2.P*A:
            print("SUCC")
            print()
        else:
            print("FAIL")
            print()
    

def test_bring():

    Hz = parse("""
    11111.........................
    1....1111.....................
    .1.......1111.................
    ..1..1.......111..............
    ...1..1..1......11............
    ....1.....1..1....11..........
    ...........1....1...111.......
    ............1.....1.1..11.....
    ..............1....1...1.11...
    .......1.......1.........1.11.
    ........1........1...1.....1.1
    ......................1.1.1.11
    """)
    Hx = parse("""
    11......1..1.........1........
    .....11..11..1................
    ...11...........1.1.1.........
    .........1..1....1......1....1
    ....................11.1.1.1..
    ..........11.......1..1...1...
    .............1.1..1.....1...1.
    ..11...........1.1.........1..
    .....1..1.....1...........1..1
    .11.........1.1........1......
    1...1..1...........1.....1....
    ......11........1.....1.....1.
    """)
    Hz = remove_dependent(Hz)
    Hx = remove_dependent(Hx)

    Lx = parse("""
    ...........11........1.1..1..1
    ..............11.......11..1.1
    ................1111.1..1.1111
    ..................11.111...11.
    .....................11..11.11
    .......................11.1111
    .........................11.1.
    ...........................111
    """)
    Lz = parse("""
    .....1....1.....11..1...1.111.
    .........11.....1111.....1.11.
    ..........1..1.11.11.....1.11.
    ......1......1..1.1...........
    ..........1..1..1.1.....1.1.11
    ...............1.1......1.....
    ................1..11...111.11
    .................11.1...1..1..
    """)

    # fixes 6 qubits
    perm = (0, 1, 11, 21, 8, 19, 25, 7, 4, 23, 14, 2, 12,
        26, 10, 22, 27, 20, 29, 5, 17, 3, 15, 9, 24, 6, 13, 16, 28, 18)

    # fixes 2 qubits
    #perm = (0, 4, 19, 25, 7, 11, 21, 8, 1, 20, 16, 3, 18, 
    #    22, 10, 26, 27, 23, 28, 6, 15, 2, 17, 13, 24, 5, 9, 14, 29, 12)

    find_fold(Hx, Hz, Lx, Lz, perm, idxs=(0,1,1,0,1,0))


def test_toric():
    """
    Moussa transverse S gate on 8-qubit toric code

    qubits are numbered starting from 1:

    +-1-+-2-
    |   |
    3   4
    |   |
    +-5-+-6-
    |   |
    7   8
    |   |

    (I @ S @ I @ ~S @ S @ I @ ~S @ I)*CZ(1,6)*CZ(3,8)

    """
    Hz = parse("1.111... .111.1.. 1...1.11")
    Hx = parse("111...1. 11.1...1 ..1.111.")

    Lx = parse(".1...1..  ......11")
    Lz = parse("....11..  ...1...1")

    #       1  2  3  4  5  6  7  8
    perm = (5, 1, 7, 3, 4, 0, 6, 2)

    find_fold(Hx, Hz, Lx, Lz, perm, idxs=(0,1,0,1), check=True)

    

def find_fold(Hz, Hx, Lz, Lx, perm, idxs=None, check=False):

    code = CSSCode(Hz, Hx, Lz, Lx)

    if check:
        code.check()

    #Hz1 = Hz[:, perm]
    #Hx1 = Hx[:, perm]
    #code1 = CSSCode(Hx1, Hz1)

    print(code)

    #print(code.P == code1.P)
    #code.check() # big...

    showmem("find_fold")
    #return

    P = code.P
    #P1 = code1.P
    m = code.mx + code.mz
    #print( (2**m)*P == P*P ) # big...

    print("fold =", end=" ")
    fixed = []
    for i, j in enumerate(perm):
        if i < j:
            print("CZ(%d, %d) *"%(i, j), end=" ")
        elif i==j:
            print("S(%d) *"%(i,), end=" ")
            fixed.append(i)
    print()

    if S is None:
        print("no S gate found.")
        return

    if idxs is None:
        # search through all of them
        opis = [(0, 1)]*len(fixed)
        idxss = list(cross(opis))

    else:
        assert len(idxs) == len(fixed)
        idxss = [idxs]

    for idxs in idxss:

        sys.stdout.flush()

        count = 0
        fold = code.I
        for i, j in enumerate(perm):
            if i < j:
                fold *= code.make_cz(i, j)
            elif i==j:
                op = [S, Sd][idxs[count]]
                count += 1
                fold *= code.make_tensor1(op, i)
    
        print("idxs =", idxs)
        print("fold =", fold)
    
        if 0:
            lhs = fold*P
            rhs = P*fold
    
            found = lhs == rhs
            print("found:", lhs==rhs)
    
            if not found:
                continue

        for i in range(code.k):
            u = code.get_encoded(i)
            u = fold(u)
            u = u.conj().T
            print("u ", end=" ", flush=True)
            for j in range(code.k):
                v = code.get_encoded(j)
                print("v ", end=" ", flush=True)
                r = numpy.dot(v, u)
                print("%.2f+%.2fj"%(r.real, r.imag), end=" ", flush=True)
            print()

        break





def weakly_T_dual_codes(m, n, minfixed=2, trials=100):
    assert 2*m==n, (m, n)
    Hx = zeros2(1, n)
    Hx[:] = 1
    for Hz in all_nonzero_codes(m, n):
        if dot2(Hz, Hx.transpose()).sum() != 0:
            continue
        Hz1 = find_kernel(Hz)
        rows = Hz1.sum(0)
        if 0 in rows:
            continue
        if Hz.sum() != Hz1.sum():
            continue

        for trial in range(trials):
            cols = list(range(n))
            shuffle(cols)
            rows = list(range(m))
            shuffle(rows)
            fixed = [i for (i,j) in enumerate(cols) if i==j]
            #if len(fixed)>=minfixed:
            #    break
            Hz2 = Hz[:, cols]
            Hz2 = Hz2[rows, :]
            assert Hz.shape == Hz2.shape
            if eq(Hz1, Hz2):
                yield (Hz, Hz1, Hx, cols)
                break


def main_T_dual():
    assert 0, "this is not going to work..."

    n = argv.get("n", 6)
    assert n%2==0
    m = n//2

    n = 8
    #Hz = parse("1111.... 11..11.. 1.1.1.1. 11111111")
    #Hz1 = Hz
    #cols = list(range(n))

    Hz = parse("1...1.11 .1...1..  ..1.111.  ...1111.")
    cols = [5, 0, 6, 4, 3, 7, 2, 1]


    Hz1 = Hz[:, cols]
    Hx = parse("1"*n)

    #for (Hz, Hz1, Hx, cols) in weakly_T_dual_codes(m, n):
    if 1:
        print("Hz:")
        print(shortstr(Hz))
        print("Hz1:")
        print(shortstr(Hz1))
        print(cols)
        print("fixed:", [i for (i,j) in enumerate(cols) if i==j])
        print()

        code0 = CSSCode(Hz, Hx)
        #code0.check()
        print(code0)
        #print(code0.opstr(code0.P))

        code1 = CSSCode(Hz1, Hx)
        #code1.check()

        gates = [Gate.T, ~Gate.T]
        opis = [(0, 1)]*n
        count = 0
        for idxs in cross(opis):
            A = None
            for i in range(n):
                B = code0.make_tensor1(gates[idxs[i]], i)
                A = B if A is None else B*A
            Ad = None
            for i in range(n):
                B = code0.make_tensor1(gates[1-idxs[i]], i)
                Ad = B if Ad is None else B*Ad
            assert A*Ad == code0.I

            #print(code0.P == A*code0.P*Ad )
            print(code0.opstr( A*code0.P*Ad ))
    
            result =( A*code0.P == code1.P*A )
            print(int(result), end="", flush=True)
#            if result:
#                print(" SUCC!", count)
#                break
            count += 1
            #print( A*code0.P == code1.P*A )
        else:
            print(" FAIL!")


def main_clifford():

    n = 3
    space = Space(n)

    pauli = []
    for name, op in space.get_basis():
        pauli.append(op)
        pauli.append(1j*op)
        pauli.append(-op)
        pauli.append(-1j*op)

    def is_clifford(U):
        for g in pauli:
            lhs = U * g * U.inverse * g.inverse
            for h in pauli:
                if lhs==h:
                    break
            else:
                return False
        return True

    def is_clifford2(U):
        result = True
        for g in pauli:
            lhs = U * g * U.inverse * g.inverse
            if not is_clifford(lhs):
                result = False
                print("X", end="", flush=True)
            else:
                print(".", end="", flush=True)
        return result

    for trial in range(5):
        op = choice(pauli)
        assert is_clifford(op)

    make_tensor1 = space.make_tensor1
    for op in [
        make_tensor1(Gate.H, 0) * make_tensor1(Gate.H, 1),
        space.make_cz(0, 1),
        make_tensor1(Gate.S, 0) * make_tensor1(Gate.H, 1) * make_tensor1(Gate.Z, 2),
    ]:
        assert is_clifford(op)


    CCZ = space.make_ccz(0, 1, 2)
    assert not is_clifford(CCZ)
    assert is_clifford2(CCZ)


def test_clifford():
    n = 1
    space = Space(n)
    G = mulclose([Gate.Z, Gate.H])
    assert len(G) == 16
    op = Gate.I
    while 1:
        op = op*Gate.Z
        #print("Z", end="")
        if op==Gate.I:
            break
        op = op*Gate.H
        #print("H", end="")
        if op==Gate.I:
            break
    #print()

    #return

    clifford = mulclose([Gate.S, Gate.H])
    assert len(clifford) == 192
    clifford = [space.make_tensor1(A, 0) for A in clifford]
    # ?


def main_identities():

    n = 1
    space = Space(n)

    I = space.I
    X = space.make_xop([0])
    Z = space.make_zop([0])
    T = space.make_tensor1(Gate.T, 0)
    S = space.make_tensor1(Gate.S, 0)
    H = space.make_tensor1(Gate.H, 0)
    ops = {"S":S, "H":H, "Z":Z, "X":X, "Si":S.inverse}
    assert H*H==I
    keys = list(ops.keys())

    if 0:
        lhs = T * X * T.inverse
        #print(lhs.todense())
        #print(space.opstr(lhs))
    
        words = []
        found = []
        for N in range(1, 7):
          for name in cross([keys]*N):
            op = reduce(mul, [ops[c] for c in name])
            words.append((name, op))
            if op == lhs:
                print(".".join(name))
                #return
            #if op not in found:
            #    found.append(op)
            #    print(len(found))
    
        return

    n = 2
    space = Space(n)
    CZ = space.make_control(Gate.Z, 0, 1)
    CX = space.make_control(Gate.X, 0, 1)
    SWAP = space.make_control(Gate.X, 0, 1) * space.make_control(Gate.X, 1, 0)
    #print(space.opstr(SWAP))

    CS = space.make_control(Gate.S, 0, 1)
    assert CS == space.make_control(Gate.S, 1, 0)
    assert CS.inverse != CS
    assert CS * CS.inverse == space.I
    SI = space.make_tensor1(Gate.S, 0)
    IS = space.make_tensor1(Gate.S, 1)

    ops = {
        "I": space.I,
        "XI" : space.make_xop([0]), 
        "IX" : space.make_xop([1]), 
        "XX" : space.make_xop([0,1]), 
        "CZ" : space.make_cz(0, 1), 
        "SI" : SI,
        "IS" : IS,
    }

    lhs = CZ * ops["IX"] * CZ.inverse
    for name, op0 in space.get_basis():
        for op in [op0, 1j*op0, -op0, -1j*op0]:
            if op == lhs:
                print(name)

    keys = list(ops.keys())
    keys.sort()

    words = []
    for (a, b, c) in cross([keys]*3):
        if a==b or b==c:
            continue
        name = (a, b, c)
        op = ops[a]*ops[b]*ops[c]
        words.append((name, op))

    lhs = CS * ops["IX"] * CS.inverse

    for name, op in words:
        if op == lhs:
            print(name)

    return

    n = 3
    space = Space(n)

    ccz = space.make_ccz(0, 1, 2)

    ops = {
        "I": space.I,
        "X0" : space.make_xop([0]), 
        "X1" : space.make_xop([1]), 
        "X2" : space.make_xop([2]), 
        "X01" : space.make_xop([0,1]), 
        "X12" : space.make_xop([1,2]), 
        "X02" : space.make_xop([0,2]), 
        "X012" : space.make_xop([0,1,2]), 
        "CZ01" : space.make_cz(0, 1), 
        "CZ02" : space.make_cz(0, 2), 
        "CZ12" : space.make_cz(1, 2), 
        "CCZ" : ccz,
    }

    keys = list(ops.keys())
    keys.sort()

    words = []
    for (a, b, c) in cross([keys]*3):
        if a==b or b==c:
            continue
        name = (a, b, c)
        op = ops[a]*ops[b]*ops[c]
        words.append((name, op))

    N = len(words)
    print("words:", N)

    for i in range(N):
      for j in range(N):
        if i==j:
            continue
        lname, lop = words[i]
        rname, rop = words[j]
        if lname[0] == rname[0]:
            continue
        if lname[2] == rname[2]:
            continue
        if (lname+rname).count("I")>1:
            continue
        lhs, rhs = ("*".join(lname), "*".join(rname))
        if lname[0] == "CCZ" and lname[2]=="CCZ":
            #print(lhs, "?=", rhs)
            if lop == rop:
                print(lhs, "==", rhs)
    


if __name__ == "__main__":

    _seed = argv.get("seed")
    if _seed is not None:
        seed(_seed)
        numpy.random.seed(_seed)

    name = argv.next() or "test"
    fn = eval(name)
    fn()

    print("%s(): OK"%name)

