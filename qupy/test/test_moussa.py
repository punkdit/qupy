#!/usr/bin/env python3

"""
Construct transversal S gate on a folded surface code.
See: https://arxiv.org/abs/1603.02286
"""

import math
from functools import reduce
from operator import mul, matmul, add

import numpy

from qupy.argv import argv
if argv.complex64:
    from qupy import scalar
    scalar.scalar = numpy.complex64
    scalar.EPSILON = 1e-6
from qupy.abstract import Space
from qupy.dense import Qu, Gate, Vector, EPSILON
from qupy.dense import genidx, bits, is_close, on, off, scalar
from qupy.tool import cross
from qupy.test import test
from qupy.util import mulclose

r2 = math.sqrt(2)


I, X, Z, Y, S, T = Gate.I, Gate.X, Gate.Z, Gate.Y, Gate.S, Gate.T

assert X*X == I
assert Z*Z == I
assert S*S == Z
#assert (T*T).is_close(S, 1e-6)
assert T*T == S

Sd = S.dag()
assert S*Sd == I

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
        v = Qu((self.d,)*n, 'u'*n)
        if idxs is not None:
            bits = [0]*n
            for i in idxs:
                bits[i] = 1
            v[tuple(bits)] = 1.
        return v

    def make_op(self, op, idxs):
        ops = [I] * self.n
        for i in idxs:
            assert 0<=i<self.n, i
            ops[i] = op
        A = reduce(matmul, ops)
        print("make_op", op.shortstr(), A.shape)
        return A

    def make_control(self, op, tgt, src):
        print("make_control", tgt, src)
        A = op.control(tgt, src, rank=self.n)
        return A

    @classmethod
    def make_surface(self, row0, row1, col0, col1):
        keys = set()
        zops = []
        xops = []
#        for i in range(row0, row1):
#          for j in range(col0, col1):
#            if row==0 and j%2==0
            
def parse(decl):
    ops = [getattr(Gate, op) for op in decl]
    op = reduce(matmul, ops)
    return op


def main_5():

    "Moussa transverse S gate on 5-qubit surface code"

    n = 5
    basis = {}
    for decl in cross(["IXZY"]*n):
        decl = ''.join(decl)
        op = parse(decl)
        basis[decl] = op
    assert len(basis)==4**n

    ops = "ZZZII IIZZZ XIXXI IXXIX".split()
    gen = [parse(decl) for decl in ops]
    for a in gen:
      for b in gen:
        assert a*b==b*a
    G = mulclose(gen)
    assert len(G) == 2**len(gen)

    G = list(G)
    P = reduce(add, G)
    print(P.shape)
    print(P*P == 2**len(gen)*P)

    def opstr(P):
        items = []
        for k,v in basis.items():
            r = (v*P).trace()
            if abs(r)<EPSILON:
                continue
            if abs(r.real - r)<EPSILON:
                r = r.real
                if abs(int(round(r)) - r)<EPSILON:
                    r = int(round(r))
            items.append("%s*%s"%(r, k))
        s = "+".join(items) or "0"
        s = s.replace("+-", "-")
        return s

    print(opstr(P))

    A = (S @ I @ ~S @ I @ S) * (Z.control(3, 1, rank=n))

    #print(opstr(A))

    P1 = A*P*~A
    print(P1 == P)

    for a in gen:
      for b in gen:
        print(int(a*A==A*b), end=" ")
      print()

    for a in gen:
        if a*A == A*a:
            continue
        print(opstr(A*a*~A))


def main_8():

    "Moussa transverse S gate on 8-qubit toric code"

    n = 8

    def get_basis(n):
        count = 0
        for decl in cross(["IXZY"]*n):
            decl = ''.join(decl)
            op = parse(decl)
            yield decl, op
            count += 1
        assert count==4**n

    ops = "ZIZZZIII IZZZIZII ZIIIZIZZ XXXIIIXI XXIXIIIX IIXIXXXI".split()
    gen = [parse(decl) for decl in ops]
    for i,a in enumerate(gen):
      for j,b in enumerate(gen):
        assert a*b==b*a, (i,j)
    G = mulclose(gen)
    assert len(G) == 2**len(gen)

    G = list(G)
    P = reduce(add, G)
    print(P.shape)
    print(P*P == 2**len(gen)*P)

    def opstr(P):
        items = []
        for k,v in get_basis(n):
            r = (v*P).trace()
            if abs(r)<EPSILON:
                continue
            if abs(r.real - r)<EPSILON:
                r = r.real
                if abs(int(round(r)) - r)<EPSILON:
                    r = int(round(r))
            items.append("%s*%s"%(r, k))
        s = "+".join(items) or "0"
        s = s.replace("+-", "-")
        return s

    #print(opstr(P))

    CZ = lambda i,j : Z.control(i, j, rank=n)
    #A = (S @ I @ ~S @ I @ S) * (Z.control(3, 1, rank=n))
    A = (I @ S @ I @ ~S @ S @ I @ ~S @ I)
    A = A*CZ(0,5)*CZ(2,7)

    #print(opstr(A))

    P1 = A*P*~A
    print(P1 == P)

    for a in gen:
      for b in gen:
        print(int(a*A==A*b), end=" ")
      print()

    for a in gen:
        if a*A == A*a:
            continue
        print(opstr(A*a*~A))


def main_13():

    assert scalar == numpy.complex64

    d = 2


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
    v0[(0,)*n] = 1

    idxs = lattice.get_idxs([(0,0,0), (0,1,0), (0,2,0)])
    v1 = lattice.make_state(idxs)
    #print(v1.shortstr())
    #return

    make_op = lattice.make_op
    get_idxs = lattice.get_idxs
    make_control = lattice.make_control

    for opi in xops:
        B = make_op(X, opi)
        v0 = v0 + B*v0
        v1 = v1 + B*v1

    #print(v0.shortstr())
    #print(v1.shortstr())

    #v[(1,)*lattice.n] = 1.
    for opi in zops:
        A = make_op(Z, opi)
        assert A*v0 == v0
        v1 = v1 + A*v1

    for opi in zops:
        A = make_op(Z, opi)
        assert A*v1 == v1, (v1 - A*v1).norm()

    v0 = v0.normalized()
    v1 = v1.normalized()

    r = v0.dag()*v1
    assert abs(r) < EPSILON, abs(r)
    #print("v0 =", v0.shortstr())
    #print("v1 =", v1.shortstr())

    geti = lambda idx : lattice.get_idxs([tuple(int(i) for i in idx)])[0]
    A =   make_control(Z, geti('100'), geti('010'))
    A = A*make_control(Z, geti('200'), geti('020'))
    A = A*make_control(Z, geti('101'), geti('011'))
    A = A*make_control(Z, geti('210'), geti('120'))

    (S,  get_idxs([(0,0,0)]))
    (Sd, get_idxs([(0,0,1)]))
    (S,  get_idxs([(1,1,0)]))
    (Sd, get_idxs([(1,1,1)]))
    (S,  get_idxs([(2,2,0)]))

    A = A*make_op(S,  get_idxs([(0,0,0)]))
    A = A*make_op(Sd, get_idxs([(0,0,1)]))
    A = A*make_op(S,  get_idxs([(1,1,0)]))
    A = A*make_op(Sd, get_idxs([(1,1,1)]))
    A = A*make_op(S,  get_idxs([(2,2,0)]))

    # check we have a logical S gate
    assert(v0 == A*v0)
    assert(1.j*v1 == A*v1)

    if 0:
        # TODO: check commutes with stabs
        for opi in xops:
            B = make_op(X, opi)
    
        for opi in zops:
            A = make_op(Z, opi)
    


if __name__ == "__main__":

    fn = argv.next() or "main"
    fn = eval(fn)
    fn()

    print("OK")

