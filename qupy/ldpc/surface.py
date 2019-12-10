#!/usr/bin/env python3

from collections import namedtuple

import numpy
from numpy import concatenate as cat

from qupy.smap import SMap
from qupy.ldpc.solve import zeros2, shortstr, solve, dot2, array2, eq2, parse
from qupy.ldpc.css import CSSCode


class Clifford(object):
    def __init__(self, A):
        self.A = A
        m, n = A.shape
        self.shape = A.shape
        assert n%2 == 0
        self.n = n//2 # qubits

    def __str__(self):
        return str(self.A)

    def __mul__(self, other):
        if isinstance(other, Clifford):
            A = dot2(self.A, other.A)
            return Clifford(A)
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

    def __eq__(self, other):
        assert isinstance(other, Clifford)
        assert self.shape == other.shape
        return eq2(self.A, other.A)

    def __ne__(self, other):
        return not self.__eq__(other)

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

    def strop(self, op, c="*"):
        smap = self.mk_smap()
        for (i, j, k) in op:
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



def test():

    n = 4
    I = Clifford.identity(n)
    H = Clifford.hadamard(n, 0)
    assert H*H == I

    CN = Clifford.cnot(n, 0, 1)
    assert CN*CN == I

    n = 3
    trivial = CSSCode(
        Lx=parse("1.."), Lz=parse("1.."), Hx=zeros2(0, n), Hz=parse(".1. ..1"))
    print(trivial.longstr())

    repitition = CSSCode(
        Lx=parse("111"), Lz=parse("1.."), Hx=zeros2(0, n), Hz=parse("11. .11"))
    print(repitition.longstr())

    CN_01 = Clifford.cnot(n, 0, 1)
    print(CN_01)
    CN_12 = Clifford.cnot(n, 1, 2)
    encode = CN_12 * CN_01

    code = encode * trivial
    print(code.longstr())

    return


    surf = Surface()
    surf.add((0, 0), (2, 4))

    print(surf)

    x_ops, z_ops = surf.get_ops()

#    for op in x_ops:
#        print()
#        print(surf.strop(op, "X"))
#    for op in z_ops:
#        print()
#        print(surf.strop(op, "Z"))

    code = surf.get_code()
    print(code)
    print(code.longstr())



if __name__ == "__main__":

    test()




