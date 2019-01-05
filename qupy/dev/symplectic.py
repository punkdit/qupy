#!/usr/bin/env python3

"""
Find all isotropic m-dimensional
subspaces of a finite symplectic space. 
Sp(n, Z/2).
"""

from math import log2

import numpy

#from bruhat.action import Perm, Group, Equ

from qupy.tool import cross, cross_upper
from qupy.argv import argv
from qupy.smap import SMap
from qupy.ldpc.solve import span, row_reduce


def zeros(shape):
    return numpy.zeros(shape, dtype=int)


def equal(A, B):
    assert A.shape == B.shape
    A = A.tostring()
    B = B.tostring()
    assert len(A)==len(B)
    return A==B


class Cell(object):
    """
        Bruhat cell for GL(n, q).
        http://math.ucr.edu/home/baez/week188.html
    """
    def __init__(self, n, leading):
        m = len(leading)
        assert m<=n
        self.m = m
        self.n = n
        self.leading= list(leading) # leading ones
        self.A = zeros((m, n))
        stars = set((i, j) for i in range(m) for j in range(n))
        for i, j in enumerate(leading):
            for i0 in range(i):
                stars.remove((i0, j))
            for j0 in range(j+1):
                stars.remove((i, j0))
            self.A[i, j] = 1
        stars = list(stars)
        stars.sort()
        self.stars = stars # free variables

    def __str__(self):
        #return str((self.A, self.stars))
        smap = SMap()
        for i in range(self.n):
          for j in range(self.m):
            smap[j, i] = str(self.A[j, i])
        for i, j in self.stars:
            smap[i, j] = "*"
        s = str(smap)
        s = s.replace("0", '.')
        return s

    def generate(self, q):
        A = self.A
        stars = self.stars
        q = range(q)
        for vals in cross((q,)*len(stars)):
            B = A.copy()
            for i, val in enumerate(vals):
                B[stars[i]] = val
            yield B

    @classmethod
    def all_cells(cls, n, m):
        for idxs in cross_upper((range(n)), m):
            cell = cls(n, idxs)
            yield cell


if 0:
    class Space(object):
        def __init__(self, U):
            U = row_reduce(U)
            m, n = U.shape
            self.m = m
            self.n = n
            self.U = U
            self._key = (m, n, U.tostring())
    
        def __str__(self):
            return str(self.U)
    
        def __eq__(self, other):
            return self._key == other._key
    
        def __ne__(self, other):
            return self._key != other._key
    
        def __hash__(self):
            return hash(self._key)
        

#def qspan(U, q):
#    n = U.shape[1]

class Space(object):
    "row span of a matrix"
    def __init__(self, U, q=2):
        m, n = U.shape
        #scalars = list(range(1, q))
        #print(scalars, U)
        #vecs = list(((r*v)) for v in span(U) for r in scalars)
        #print(vecs)
        #vecs = list(((r*v)%q) for v in span(U) for r in scalars)
        #print(vecs)
        #print()
        #vecs = set(((r*v)%q).tostring() for v in span(U) for r in scalars)

        vecs = set()
        vals = list(range(q))
        for u in cross((vals,)*m):
            vec = numpy.dot(u, U) % q
            vecs.add(vec.tostring())

        vecs = list(vecs)
        vecs.sort()
        vecs = tuple(vecs)
        #assert len(vecs) == len(set(vecs))
        self.n = n
        self.U = U
        self._key = vecs
        N = len(vecs)
        m = int(round(log2(N)/log2(q)))
        assert q**m == N
        self.m = m # rank of U
    
    def __str__(self):
        return str(self.U)

    def __eq__(self, other):
        return self._key == other._key

    def __ne__(self, other):
        return self._key != other._key

    def __hash__(self):
        return hash(self._key)
        


def test_space():
    U = numpy.array([[0,0,1,0], [0,0,0,1]])
    V = numpy.array([[0,0,1,1], [0,0,0,1]])
    assert Space(U) == Space(V)
    V = numpy.array([[0,0,0,1], [0,0,1,0]])
    assert Space(U) == Space(V)
    V = numpy.array([[0,1,0,1], [0,0,1,0]])
    assert Space(U) != Space(V)

    q = 3
    U = numpy.array([[0,0,1,0]])
    V = numpy.array([[0,0,2,0]])
    assert Space(U, q) == Space(V, q)

test_space()



def mk_form(n, q):
    # symplectic form
    A = zeros((n, n))
    for i in range(n//2):
        A[i, i+n//2] = 1
        A[i+n//2, i] = q-1
    return A


def borel_gl(n, q):
    "Borel subgroup of GL(n, q)"

    A = numpy.zeros((n, n), dtype=int)

    stars = [(i, j) for i in range(0, n-1) for j in range(i+1, n)]

    nz = list(range(1, q))
    diags = [diag for diag in cross((nz,)*n)]
    #print(len(diags), len(stars))

    N = len(stars)
    for vals in cross((range(q),)*N):
      for diag in diags:
        X = A.copy()
        for i, val in enumerate(vals):
            X[stars[i]] = val
        for i in range(n):
            X[i, i] = diag[i]
        yield X


def borel_sp_slow(n, q):
    "Borel subgroup of Sp(n, q)"

    J = mk_form(n, q)
    for X in borel_gl(n, q):
        XtJX = numpy.dot(X.transpose(), numpy.dot(J, X)) % q
        if XtJX.tostring() == J.tostring():
            print(X)
            yield X


def borel_sp(n, q):
    "Borel subgroup of Sp(n, q)"

    assert n%2==0
    n2 = n//2

    J = mk_form(n, q)

    mul = list(range(1, q))
    inv = {}
    for i in mul:
        for j in mul:
            if (i*j)%q == 1:
                inv[i] = j
                break
        else:
            assert 0

    diags = []
    for diag0 in cross((mul,)*n2):
        diag1 = tuple(inv[i] for i in diag0)
        diags.append(diag0 + diag1)
        #print(diag0 + diag1)

    stars = [(i, j) for i in range(0, n2) for j in range(n2, n)]

    J = mk_form(n, q)
    N = len(stars)
    X = numpy.zeros((n, n), dtype=int)
    for vals in cross((range(q),)*N):
        for i, star in enumerate(stars):
            X[star] = vals[i]
        for diag in diags:
            for i in range(n):
                X[i, i] = diag[i]
            XtJX = numpy.dot(X.transpose(), numpy.dot(J, X)) % q
            if XtJX.tostring() == J.tostring():
                yield X.copy()
        
    


def test_borel():

    n = argv.get("n", 4)
    m = argv.get("m", 1)
    q = argv.get("q", 2)

    found = 0
    for g in borel_sp(n, q):
        found += 1
    print("found:", found)

    return

    for cell in Cell.all_cells(n, m):
        print(cell)
        for v in cell.generate(q):
            print(v)
        print()




def find():

    n = argv.get("n", 4)
    assert n%2 == 0, repr(n)
    m = argv.get("m", 2)
    q = argv.get("q", 2)

    A = mk_form(n, q)
    
    count = 0
    for cell in Cell.all_cells(n, m):
        found = 0
        for U in cell.generate(q):
            B = numpy.dot(U, numpy.dot(A, U.transpose())) % 2
            if B.max():
                continue
            count += 1
            found += 1
        print(cell)
        print(found)
        print()
    print(count)


def bruhat():

    n = argv.get("n", 4)
    assert n%2 == 0, repr(n)
    m = argv.get("m", 2)
    q = argv.get("q", 2)

    # symplectic form
    A = mk_form(n, q)
    
    # all non-zero vectors
    vals = list(range(q))
    vecs = list(cross((vals,)*n))
    assert sum(vecs[0])==0
    vecs.pop(0)
    
    # find unique spaces
    spaces = set()
    for U in cross_upper(vecs, m):
        U = numpy.array(U)
        U.shape = m, n
        B = numpy.dot(U, numpy.dot(A, U.transpose())) % q
        if B.max():
            continue

        space = Space(U, q)
        if space.m != m:
            continue
        spaces.add(space)

        if 0:
            #space = [str(v) for v in span(U)] # SLOW
            space = [v.tostring() for v in span(U)] # q==2 only
            if len(space) != q**m:
                continue
            space.sort()
            #space = ''.join(space)
            #print(space)
            space = tuple(space)
            spaces.add(space)
    
    N = len(spaces)
    print("points:", N)
    if argv.verbose:
        for X in spaces:
            print(X)

    B = list(borel_sp(n, q))
    print("borel:", len(B))
    assert len(B)

    spaces = list(spaces)
    lookup = dict((space, i) for (i, space) in enumerate(spaces))

    orbits = list(set([space]) for space in spaces)

    perms = []
    for g in B:
        perm = []
        for i, space in enumerate(spaces):
            U = numpy.dot(space.U, g) % q
            t = Space(U, q)
            #if t not in lookup:
            #    print(space)
            #    print(t)
            perm.append(lookup[t])
        perms.append(perm)
        print(".", end=" ", flush=True)
    print()

    remain = set(range(N))
    orbits = []
    while remain:
        i = iter(remain).__next__()
        remain.remove(i)
        orbit = [i]
        for perm in perms:
            j = perm[i]
            if j in remain:
                remain.remove(j)
                orbit.append(j)
        orbits.append(orbit)

    orbits.sort(key = len)
    print("%d orbits:" % len(orbits))
    for orbit in orbits:
        print("size =", len(orbit))
        for idx in orbit:
            space = spaces[idx]
            U = space.U
            if q==2:
                U = row_reduce(U)
            #print(U)



if __name__ == "__main__":

    fn = argv.next()
    if fn is not None:
        fn = eval(fn)
        fn()
 


