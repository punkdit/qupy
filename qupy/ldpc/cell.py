#!/usr/bin/env python3

"""
another version: bruhat/morse.py
"""


from collections import namedtuple
from functools import reduce
from operator import mul
from random import shuffle, seed

import numpy
from numpy import concatenate as cat

from qupy.util import mulclose_fast
from qupy.argv import argv
from qupy.smap import SMap
from qupy.ldpc.solve import zeros2, shortstr, parse, pseudo_inverse, int_scalar, dot2
from qupy.ldpc.css import CSSCode
#from qupy.ldpc.decoder import StarDynamicDistance

from qupy.argv import argv
from qupy.smap import SMap

try:
    from pyx import canvas, path, deco, trafo, style, text, color, deformer, unit
    from pyx.color import rgb
    
    black = rgb(0., 0., 0.)
    blue = rgb(0.1, 0.1, 0.9)
    red = lred = rgb(1., 0., 0.)
    green = rgb(0.0, 0.6, 0.0)
    white = rgb(1., 1., 1.)
    #shade = rgb(0.75, 0.55, 0)
    grey = rgb(0.75, 0.75, 0.75)
    yellow = rgb(1., 1., 0.)
    
    st_dashed = [style.linestyle.dashed]
    st_dotted = [style.linestyle.dotted]
    st_round = [style.linecap.round]
    
    st_thick = [style.linewidth.thick]
    st_Thick = [style.linewidth.Thick]
    st_THick = [style.linewidth.THick]
    st_THICK = [style.linewidth.THICK]
    #print dir(style.linewidth)

    import boxs

except ImportError:
    print("warning: pyx not found")



class Matrix(object):
    """ rows and cols : list of hashable items """

    def __init__(self, rows, cols, elements={}):
        rows = list(rows)
        cols = list(cols)
        self.rows = rows
        self.cols = cols
        self.set_rows = set(rows)
        self.set_cols = set(cols)
        #self.row_lookup = dict((row, idx) for (idx, row) in rows)
        #self.col_lookup = dict((col, idx) for (idx, col) in cols)
        self.elements = dict(elements)

    def copy(self):
        return Matrix(self.rows, self.cols, self.elements)

    def __len__(self):
        return len(self.rows)

    def __str__(self):
        return "Matrix(%d, %d)"%(len(self.rows), len(self.cols))

    def __eq__(self, other):
        return self.elements == other.elements

    def __ne__(self, other):
        return self.elements != other.elements

    @classmethod
    def identity(cls, items):
        elements = dict(((i, i), 1) for i in items)
        A = cls(items, items, elements)
        return A

    @classmethod
    def inclusion(cls, rows, cols):
        A = cls(rows, cols)
        for i in cols:
            A[i, i] = 1
        return A

    @classmethod
    def retraction(cls, rows, cols):
        A = cls(rows, cols)
        for i in rows:
            A[i, i] = 1
        return A

    def __setitem__(self, key, value):
        row, col = key
        #idx = self.row_lookup[row]
        #jdx = self.col_lookup[col]
        assert row in self.set_rows
        assert col in self.set_cols
        elements = self.elements
        if value != 0:
            elements[row, col] = value
        elif (row, col) in elements:
            del elements[row, col]

    def __getitem__(self, item):
        row, col = item
        
        if isinstance(row, slice) and isinstance(col, slice):
            assert row == slice(None)
            assert col == slice(None)
            value = self

        elif isinstance(col, slice):
            assert col == slice(None)
            value = []
            assert row in self.set_rows
            for (r,c) in self.elements.keys():
                if r == row:
                    value.append(c)

        elif isinstance(row, slice):
            assert row == slice(None)
            value = []
            assert col in self.set_cols
            for (r,c) in self.elements.keys():
                if c == col:
                    value.append(r)

        else:
            #idx = self.row_lookup[row]
            #jdx = self.col_lookup[col]
            assert row in self.set_rows
            assert col in self.set_cols
            value = self.elements.get((row, col), 0)
        return value

    def __add__(self, other):
        assert self.set_cols == other.set_rows
        A = self.copy()
        # could optimize, but do we care...
        for i,j in other.elements.keys():
            A[i, j] += other[i, j]
        return A

    def __mul__(self, other):
        assert self.set_cols == other.set_rows
        A = Matrix(self.rows, other.cols)
        # could optimize, but do we care...
        #for i in self.rows:
        #  for j in self.cols:
        for (i, j) in self.elements.keys():
            for k in other.cols:
                A[i, k] = A[i, k] + self[i, j] * other[j, k]
        return A

    def sum(self):
        return sum(self.elements.values())

    def todense(self, rows=None, cols=None):
        if rows is None:
            rows = self.rows
        if cols is None:
            cols = self.cols
        assert set(rows) == self.set_rows, "%s\n%s"%(set(rows), self.set_rows)
        assert set(cols) == self.set_cols, "%s\n%s"%(set(cols), self.set_cols)
        row_lookup = dict((row, idx) for (idx, row) in enumerate(rows))
        col_lookup = dict((col, idx) for (idx, col) in enumerate(cols))
        A = numpy.zeros((len(rows), len(cols)), dtype=int_scalar)
        elements = self.elements
        for (row, col) in elements.keys():
            value = elements[row, col]
            A[row_lookup[row], col_lookup[col]] = value
        return A


class Cell(object):

    def __init__(self, grade, bdy=[], key=None):
        self.grade = grade
        self.bdy = list(bdy)
        self.key = key

    def __repr__(self):
        return "Cell(%d, %s, %s)"%(self.grade, self.bdy, self.key)

    def __str__(self):
        return "Cell(%d, %s)"%(self.grade, self.key)

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

    def __getitem__(self, idx):
        return self.bdy[idx]

    def __lt__(self, other):
        return self.key < other.key

    def __le__(self, other):
        return self.key <= other.key

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

    def get_cells(self, grade):
        cells = list(self.lookup[grade].values())
        cells.sort()
        return cells

#    def __XX__get_code(self):
#        verts = self.lookup[0]
#        edges = self.lookup[1]
#        faces = self.lookup[2]
#            
#        n = len(edges)
#        mx = len(verts)
#        mz = len(faces)
#        Hx = zeros2(mx, n)
#        Hz = zeros2(mz, n)
#
#        cols = list(edges.keys())
#        cols.sort()
#        rows = list(verts.keys())
#        rows.sort()
#
#        for j, col in enumerate(cols):
#          edge = edges[col]
#          for i, row in enumerate(rows): # ugh
#            vert = verts[row]
#            if vert in edge.bdy:
#                Hx[i, j] = 1
#        
#        rows = list(faces.keys())
#        rows.sort()
#
#        for j, col in enumerate(cols):
#          edge = edges[col]
#          for i, row in enumerate(rows): # ugh
#            face = faces[row]
#            if edge in face.bdy:
#                Hz[i, j] = 1
#
#        #print()
#        #print(shortstr(Hx))
#        #print()
#        #print(shortstr(Hz))
#
#        code = CSSCode(Hx=Hx, Hz=Hz)
#        return code
        
    def get_bdymap(self, grade):
        # grade -> grade-1
        cols = self.get_cells(grade)
        rows = self.get_cells(grade-1)
        A = Matrix(rows, cols)
        for col in cols:
            assert col.grade == grade
            for row in col:
                assert row.grade == grade-1
                A[row, col] = (A[row, col] + 1)%2
        return A

    def get_parity_checks(self, grade=1):
        verts = self.get_cells(grade-1)
        edges = self.get_cells(grade)
        faces = self.get_cells(grade+1)
            
        Hzt = self.get_bdymap(grade+1)
        Hzt = Hzt.todense(edges, faces)
        Hz = Hzt.transpose()

        Hx = self.get_bdymap(grade)
        Hx = Hx.todense(verts, edges)
        return Hz, Hx

    def get_code(self, grade=1):
        Hz, Hx = self.get_parity_checks(grade)
        code = CSSCode(Hx=Hx, Hz=Hz)
        return code
        
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
        for i in range(i0, i1-1):
          for j in range(j0, j1-1):
            top = edges.get((i, j, "h"))
            left = edges.get((i, j, "v"))
            bot = edges.get((i+1, j, "h"))
            right = edges.get((i, j+1, "v"))
            bdy = [top, left, bot, right]
            bdy = [cell for cell in bdy if cell]
            if len(bdy)==0:
                continue
            cell = Cell(2, bdy)
            self.set((i, j), cell)

    def build_torus(self, rows, cols):

        # build verts
        for i in range(rows):
          for j in range(cols):
            cell = Cell(0)
            key = (i, j)
            self.set(key, cell)

        verts = self.lookup[0]

        # build edges
        for i in range(rows):
          for j in range(cols):
            ks = "hv" # horizontal, _vertical edge
            for k in ks:
                bdy = []
                vert = verts.get((i, j))
                assert vert is not None
                bdy.append(vert)
                if k == "h":
                    vert = verts.get((i, (j+1)%cols))
                else:
                    vert = verts.get(((i+1)%rows, j))
                assert vert is not None
                bdy.append(vert)
                assert len(bdy)==2
                cell = Cell(1, bdy)
                key = (i, j, k)
                self.set(key, cell)

        edges = self.lookup[1]

        # build faces
        for i in range(rows):
          for j in range(cols):
            top = edges.get((i, j, "h"))
            left = edges.get((i, j, "v"))
            bot = edges.get(((i+1)%rows, j, "h"))
            right = edges.get((i, (j+1)%cols, "v"))
            bdy = [top, left, bot, right]
            bdy = [cell for cell in bdy if cell]
            assert len(bdy)==4
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

    def get_smap(self):
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
        return smap

    def __str__(self):
        smap = self.get_smap()
        return str(smap)


class Flow(object):
    def __init__(self, cx):
        self.cx = cx
        self.pairs = {}

    def add(self, src, tgt):
        assert isinstance(src, Cell)
        assert isinstance(tgt, Cell)
        assert src.grade == tgt.grade-1
        assert src in tgt
        pairs = self.pairs.setdefault(src.grade, [])
        pairs.append((src, tgt))

    def remove(self, src, tgt):
        self.pairs[src.grade].remove((src, tgt))

    def get_pairs(self):
        pairs = []
        cx = self.cx
        edges = cx.get_cells(1)
        faces = cx.get_cells(2)
        for edge in edges:
            for vert in edge:
                pairs.append((vert, edge))
        for face in faces:
            for edge in face:
                pairs.append((edge, face))
        return pairs

    def get_critical(self, grade):
        cx = self.cx
        remove = set()
        for _grade, pairs in self.pairs.items():
          for src, tgt in pairs:
            remove.add(src)
            remove.add(tgt)
        critical = []
        cells = cx.get_cells(grade)
        for cell in cells:
            if cell not in remove:
                critical.append(cell)
        return critical

    def get_flowmap(self, grade):
        # grade --> grade+1
        cx = self.cx
        edges = cx.get_cells(grade)
        faces = cx.get_cells(grade+1)
        A = Matrix(faces, edges)
        pairs = self.pairs.setdefault(grade, [])
        for src, tgt in pairs:
            assert src.grade == grade
            A[tgt, src] += 1
        return A

#    def get_adj(self, grade):
#        "return flow (adjacency) matrix for this grade"
#        cells = self.cx.get_cells(grade)
#        lookup = dict((cell, idx) for (idx, cell) in enumerate(cells))
#        N = len(cells)
#        A = numpy.zeros((N, N), dtype=int_scalar)
#        pairs = self.pairs[grade-1]
#        flow = dict((src, tgt) for src, tgt in pairs)
#        for i, cell in enumerate(cells):
#            for src in cell.bdy:
#                tgt = flow.get(src)
#                if tgt != None:
#                    j = lookup[tgt]
#                    if i!=j:
#                        A[j, i] = 1
#        return A, lookup

    def get_adj(self, grade):
        "return flow (adjacency) matrix for this grade"
        cx = self.cx
        A = cx.get_bdymap(grade) # grade --> grade-1
        B = self.get_flowmap(grade-1) # grade-1 --> grade
        C = B*A # grade --> grade
        for cell in cx.get_cells(grade):
            C[cell, cell] = 0
        return C

    def accept(self, src, tgt):
        assert isinstance(src, Cell)
        assert isinstance(tgt, Cell)
        assert src.grade == tgt.grade-1
        pairs = self.pairs.setdefault(src.grade, [])
        for pair in pairs:
            if pair[0] == src or pair[1] == tgt:
                return False
        pairs = self.pairs.setdefault(src.grade-1, [])
        for pair in pairs:
            if pair[1] == src:
                return False
        pairs = self.pairs.setdefault(src.grade+1, [])
        for pair in pairs:
            if pair[0] == tgt:
                return False

        cx = self.cx
        self.add(src, tgt)
        for grade in (1, 2):
            A = self.get_adj(grade)
            #print(shortstr(A))
            #N = len(A)
            B = A.copy()
            cells = cx.get_cells(grade)
            while B.sum():
                #B = numpy.dot(A, B)
                B = A*B
                #for j in range(N):
                for j in cells:
                    if B[j, j]:
                        self.remove(src, tgt)
                        return False
        self.remove(src, tgt)
        return True

    def __str__(self):
        items = ['%s->%s' % (src.key, tgt.key) for (src, tgt) in self.pairs[0]]
        items += ['%s->%s' % (src.key, tgt.key) for (src, tgt) in self.pairs[1]]
        s = ", ".join(items)
        return "Flow(%s)"%(s,)

    def build(self):
        pairs = self.get_pairs()
        shuffle(pairs)
    
        idx = 0
        while idx < len(pairs):
            src, tgt = pairs[idx]
            if self.accept(src, tgt):
                self.add(src, tgt)
            idx += 1

    def morse_homology(self, grade=1):
        cx = self.cx
        cells = {}
        crit = {}
        for grade in [2, 1, 0]:
            cells[grade] = cx.get_cells(grade)
            crit[grade] = self.get_critical(grade)

        chain = []
        for grade in [2, 1]:
            bdy = Matrix(crit[grade-1], crit[grade])
            A = self.get_adj(grade)  # grade --> grade-1 --> grade
            B = cx.get_bdymap(grade) # grade --> grade-1
            C = Matrix.identity(cells[grade])
            while C.sum():
                #print()
                #print(shortstr(C.todense()))
                D = B*C # grade --> grade-1
                for a in crit[grade-1]:
                  for b in crit[grade]:
                    if D[a, b]:
                        bdy[a, b] += D[a, b]

                C = A*C

            #print(bdy.todense())
            chain.append(bdy)
        return chain

    def render(self, cvs=None, name=None, size=1.0, double=False):
        if cvs is None:
            cvs = canvas.canvas()

        dx, dy = size, -size
        r = 0.05 * size
        mul = 1.5 # critical radius

        cx = self.cx
        critical = self.get_critical(0)
        cells = cx.get_cells(0)
        for cell in cells:
            i, j = cell.key
            x, y = dx*j, dy*i
            cell.pos = (x, y)
            if cell in critical:
                cvs.stroke(path.circle(x, y, mul*r), [blue]+st_thick)
            else:
                cvs.fill(path.circle(x, y, r))

        critical = self.get_critical(1)
        cells = cx.get_cells(1)
        for cell in cells:
            i, j, k = cell.key
            assert k in "hv"
            x0, y0 = dx*j, dy*i
            if k=="h":
                x1, y1 = dx*(j+1), dy*i
            else:
                x1, y1 = dx*j, dy*(i+1)

            x = 0.5*(x0+x1)
            y = 0.5*(y0+y1)
            if cell in critical:
                cvs.stroke(path.circle(x, y, mul*r), [red]+st_thick)
            else:
                cvs.fill(path.circle(x, y, 1*r), [red]+st_thick)
            cvs.stroke(path.line(x0, y0, x1, y1))
            cell.pos = (x0, y0, x1, y1, x, y)

        critical = self.get_critical(2)
        cells = cx.get_cells(2)
        for cell in cells:
            i, j = cell.key
            x, y = dx*(j+0.5), dy*(i+0.5)
            cell.pos = (x, y)
            if cell in critical:
                cvs.stroke(path.circle(x, y, mul*r), [green]+st_thick)

        for src, tgt in self.pairs[0]:
            i0, j0 = src.key # vert
            i1, j1, k = tgt.key # edge
            if k=="h":
                assert i0 == i1 # same row
                if j0==j1:
                    pass
                elif j0-1==j1:
                    pass
                elif j0<j1:
                    j0 = j1+1
            else:
                assert j0 == j1
                if i1>i0:
                    i0 = i1+1

            x0, y0 = dx*j0, dy*i0
            if k=="h":
                x1, y1 = dx*(j1+0.5), dy*i1
            else:
                x1, y1 = dx*j1, dy*(i1+0.5)
            if double:
                dx1, dy1 = x1-x0, y1-y0
                x1 += dx1
                y1 += dy1
            cvs.stroke(path.line(x0, y0, x1, y1), [deco.earrow(), green]+st_THick)

        for src, tgt in self.pairs[1]:
            i0, j0, k = src.key # edge
            i1, j1 = tgt.key # face

            if k=="h":
                assert j0==j1 # same col
                if i0==i1 or i0-1==i1:
                    pass
                elif i0 < i1:
                    #i0 = i1+1
                    i1 = i0-1
            if k=="v":
                assert i0==i1 # same row
                if j0==j1 or j0-1==j1:
                    pass
                elif j0 < j1:
                    #j0 = j1+1
                    j1 = j0-1

            x0, y0 = dx*j0, dy*i0
            if k=="h":
                x1, y1 = dx*(j0+1), dy*i0
            else:
                x1, y1 = dx*j0, dy*(i0+1)
            x0 = 0.5*(x0+x1)
            y0 = 0.5*(y0+y1)
            x1, y1 = dx*(j1+0.5), dy*(i1+0.5)

            if double:
                dx1, dy1 = x1-x0, y1-y0
                x0 -= dx1
                y0 -= dy1
            cvs.stroke(path.line(x0, y0, x1, y1), [deco.earrow(), blue]+st_THick)

        if name:
            cvs.writePDFfile(name)
            cvs.writeSVGfile(name)
            print("saved", name)

        return cvs



def render_flow():
    m, n = argv.get("m", 3), argv.get("n", 3) 
    nrows, ncols = argv.get("nrows", 1), argv.get("ncols", 1) 
    double = argv.double
    name = argv.get("name", "output")

    cx = Complex()

    if argv.torus:
        cx.build_torus(m, n)
        signature = (1, 2, 1)

    elif argv.disc:
        cx.build_surface((0, 0), (m+1, n+1))
        signature = (1, 0, 0)

    else:
    
        cx.build_surface((0, 0), (m+1, n+1), open_top=True, open_bot=True)
        signature = (0, 1, 0)

    if argv.showcode:
        code = cx.get_code()
        print(code)
        return

    rows = []
    row = []
    while len(rows) < nrows:
        flow = Flow(cx)
        flow.build()
        
        dims = [len(flow.get_critical(grade)) for grade in [0, 1, 2]]
        print("dims:", dims)
        if signature is not None and tuple(dims) != signature:
            continue

        #flow.render(name="output")
        row.append(flow.render(size=1.3, double=double))
        if len(row)==ncols:
            rows.append(row)
            row = []
        print(len(row), len(rows))

    PAD = 2.0
    rows = [boxs.HBox([cvs for cvs in row], pad=PAD) for row in rows]
    box = boxs.VBox(rows, pad=PAD)
    box.save(name)

    print("OK")


def test_flow():

    seed(3)

    m, n = argv.get("m", 3), argv.get("n", 3) 
    cx = Complex()

    if argv.disc:
        cx.build_surface((0, 0), (m, n))
        k = 0
    elif argv.surface:
        cx.build_surface((0, 0), (m, n), open_top=True, open_bot=True)
        k = 0
    elif argv.torus:
        cx.build_torus(m, n)
        k = 2
    else:
        cx.build_torus(m, n)
        k = 2
    
    #print(cx)

    for trial in range(10):

        flow = Flow(cx)
        flow.build()

        if argv.render:
            flow.render(name="output")
            break

        chain = flow.morse_homology(1)
        A, B = chain
        BA = B*A
        for key,v in BA.elements.items():
            assert v%2 == 0
        #print(BA.todense())

        Hzt = A.todense() % 2
        Hz = Hzt.transpose()
        Hx = B.todense() % 2

        assert dot2(Hx, Hzt).sum() == 0
        code = CSSCode(Hx=Hx, Hz=Hz)
        #print("k =", code.k)
        assert code.k == k, (code.k, k)




def test_surface():

    rows, cols = 4, 3

    cx = Complex()
    cx.build_surface((0, 0), (rows, cols))
    edges = cx.lookup[1].values()
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

    cx = Complex()
    cx.build_surface((0, 0), (rows, cols), 
        open_top=True, open_bot=True, open_left=True, open_right=True)
    code = cx.get_code()
    assert code.n == 7
    assert code.k == 0
    assert code.mx == 2
    assert code.mz == 5

    cx = Complex()
    cx.build_torus(rows, cols)
    #print(cx)
    code = cx.get_code()
    #print(code)



if __name__ == "__main__":

    name = argv.next()

    if argv.profile:
        import cProfile as profile
        profile.run("%s()"%name)

    elif name:

        fn = eval(name)
        fn()

    else:

        test_surface()
        test_flow()

