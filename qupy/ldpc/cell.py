#!/usr/bin/env python3


from collections import namedtuple
from functools import reduce
from operator import mul

import numpy
from numpy import concatenate as cat

from qupy.util import mulclose_fast
from qupy.argv import argv
from qupy.smap import SMap
from qupy.ldpc.solve import zeros2, shortstr, parse, pseudo_inverse
from qupy.ldpc.css import CSSCode
#from qupy.ldpc.decoder import StarDynamicDistance

from qupy.argv import argv
from qupy.smap import SMap




class Cell(object):

    def __init__(self, grade, bdy=[], key=None):
        self.grade = grade
        self.bdy = list(bdy)
        self.key = key

    def __str__(self):
        return "Cell(%d, %s, %s)"%(self.grade, self.bdy, self.key)
    __repr__ = __str__

    def check(self):
        grade = self.grade
        for cell in self.bdy:
            assert cell.grade == grade-1

    def append(self, cell):
        assert isinstance(cell, Cell)
        assert cell.grade == self.grade-1
        self.bdy.append(cell)

    def __len__(self):
        return len(self.bdy)

    #def get_dual(self):



class Complex(object):

    def __init__(self):
        self.lookup = {0:{}, 1:{}, 2:{}}

    def set(self, key, cell):
        assert isinstance(cell, Cell)
        lookup = self.lookup
        cells = lookup.setdefault(cell.grade, {})
        assert cells.get(key) is None
        cells[key] = cell
        assert cell.key is None
        cell.key = key

    def build_surface(self, top_left, bot_right, 
        open_top=False, open_bot=False, open_left=False, open_right=False):
        # open boundary is "rough". default to smooth

        i0, j0 = top_left
        i1, j1 = bot_right

        # build verts
        for i in range(i0, i1):
          for j in range(j0, j1):
            if i==i0 and open_top:
                continue
            if i==i1-1 and open_bot:
                continue
            if j==j0 and open_left:
                continue
            if j==j1-1 and open_right:
                continue
            cell = Cell(0)
            key = (i, j)
            self.set(key, cell)

        verts = self.lookup[0]

        # build edges
        for i in range(i0, i1):
          for j in range(j0, j1):
            ks = "hv" # horizontal, _vertical edge
            if i==i1-1 and j==j1-1:
                continue
            elif i==i1-1:
                ks = "h"
            elif j==j1-1:
                ks = "v"
            for k in ks:
                bdy = []
                vert = verts.get((i, j))
                if vert is not None:
                    bdy.append(vert)
                if k == "h":
                    vert = verts.get((i, j+1))
                else:
                    vert = verts.get((i+1, j))
                if vert is not None:
                    bdy.append(vert)
                if len(bdy)==0:
                    continue
                cell = Cell(1, bdy)
                key = (i, j, k)
                self.set(key, cell)

        edges = self.lookup[1]

        # build faces
        for i in range(i0, i1):
          for j in range(j0, j1):
            top = edges.get((i, j, "h"))
            left = edges.get((i, j, "v"))
            bot = edges.get((i+1, j, "h"))
            right = edges.get((i, j+1, "v"))
            bdy = [top, left, bot, right]
            bdy = [cell for cell in bdy if cell]
            if len(bdy)<3:
                continue
            cell = Cell(2, bdy)
            self.set((i, j), cell)

    def get_coord(self, i, j, k=None):
        mul = 2
        if k is None:
            row = mul*i
            col = mul*j
        else:
            assert k in "hv"
            k = "hv".index(k)
            row = mul*i + k
            col = mul*j + (1-k)
        return row, col

    def __str__(self):
        get_coord = self.get_coord
        smap = SMap()

        verts = self.lookup[0]
        edges = self.lookup[1]
        faces = self.lookup[2]
        for key in verts.keys():
            i, j = key
            row, col = get_coord(i, j)
            smap[row, col] = "."

        for key in edges.keys():
            i, j, k = key
            row, col = get_coord(i, j, k)
            c = {"h":"-", "v":"|"}[k]
            smap[row, col] = c

        for key in faces.keys():
            i, j = key
            row, col = get_coord(i, j)
            smap[row+1, col+1] = "o"

        return str(smap)

    def get_code(self):
        verts = self.lookup[0]
        edges = self.lookup[1]
        faces = self.lookup[2]
            
        n = len(edges)
        mx = len(verts)
        mz = len(faces)
        Hx = zeros2(mx, n)
        Hz = zeros2(mz, n)

        cols = list(edges.keys())
        cols.sort()
        rows = list(verts.keys())
        rows.sort()

        for j, col in enumerate(cols):
          edge = edges[col]
          for i, row in enumerate(rows): # ugh
            vert = verts[row]
            if vert in edge.bdy:
                Hx[i, j] = 1
        
        rows = list(faces.keys())
        rows.sort()

        for j, col in enumerate(cols):
          edge = edges[col]
          for i, row in enumerate(rows): # ugh
            face = faces[row]
            if edge in face.bdy:
                Hz[i, j] = 1

        #print()
        #print(shortstr(Hx))
        #print()
        #print(shortstr(Hz))

        code = CSSCode(Hx=Hx, Hz=Hz)
        return code
        


def test_cell():

    top = Cell(2)

    top.append(Cell(1))

    rows, cols = 4, 3

    cx = Complex()
    cx.build_surface((0, 0), (rows, cols))
    #print(cx)
    edges = cx.lookup[1].values()
    #for cell in edges:
    #    print(cell.key, cell.bdy)
    for cell in edges:
        assert len(cell)==2, cell
    code = cx.get_code()
    assert code.n == 17
    assert code.k == 0
    assert code.mx == 11
    assert code.mz == 6

    cx = Complex()
    cx.build_surface((0, 0), (rows, cols), open_top=True, open_bot=True)
    code = cx.get_code()
    assert code.n == 13
    assert code.k == 1
    assert code.mx == 6
    assert code.mz == 6

    cx = Complex()
    cx.build_surface((0, 0), (rows, cols), open_left=True, open_right=True)
    code = cx.get_code()
    assert code.n == 11
    assert code.k == 1
    assert code.mx == 4
    assert code.mz == 6



if __name__ == "__main__":

    name = argv.next()
    if name:

        fn = eval(name)
        fn()

    else:

        test_cell()

