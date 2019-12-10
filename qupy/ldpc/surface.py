#!/usr/bin/env python3

from collections import namedtuple

import numpy

from qupy.smap import SMap
from qupy.ldpc.solve import zeros2, shortstr, solve, dot2, array2
from qupy.ldpc.css import CSSCode


#_Coord = namedtuple("_Coord", ["i", "j", "k"])
#class Coord(_Coord):
#
#    def __new__(cls, i, j, k):
#        assert k==0 or k==1
#        return _Coord.__new__(cls, i, j, k)
#
#    #def __add__(self, 
#
#    def __lt__(self, other):
#
#        assert isinstance(other, Coord)
#        if self.i < other.i and self.j < other.j:
#            return True
#        if self.i < other.i and self.j <= other.j:
#            return True
#        if self.i <= other.i and self.j < other.j:
#            return True
#        if self.i <= other.i and self.j <= other.j and self.k < other.k:
#            return True
#        return False
#
#    def __le__(self, other):
#        assert isinstance(other, Coord)
#        return self==other or self.__lt__(other)
#
#    def span(self, other):
#        if not self<other:
#            return
#        for i in range(self.i, other.i):
#          for j in range(self.j, other.j):
#
#print(Coord(0, 0, 0))


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




