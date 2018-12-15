#!/usr/bin/env python3

"""
Fooling around with polynomial invariants of
finite subgroups of SU(2).
"""

from math import sqrt
#from random import choice, randint, seed, shuffle

from functools import reduce
from operator import mul, matmul, add

import numpy
from numpy import exp, pi

from qupy import scalar
#scalar.use_reals()
from qupy.scalar import EPSILON, MAX_GATE_SIZE
from qupy.dense import Qu
from qupy.util import mulclose, show_spec
from qupy.tool import fstr
from qupy.argv import argv

from qupy.dev.comm import Poly


def cyclotomic(n):
    return numpy.exp(2*numpy.pi*1.j/n)


def build():

    global Octa, Tetra, Icosa, Sym, Pauli, RealPauli, RealCliff, Cliff

    # ----------------------------------
    #
    
    gen = [
        [[0, 1], [1, 0]],
    ]
    gen = [Qu((2, 2), 'ud', v) for v in gen]

    Sym = mulclose(gen)
    assert len(Sym)==2


    # ----------------------------------
    #

    gen = [
        [[0, 1], [1, 0]],
        [[1, 0], [0, -1]],
    ]
    gen = [Qu((2, 2), 'ud', v) for v in gen]

    RealPauli = mulclose(gen)
    assert len(RealPauli)==8

    # ----------------------------------
    #

    r = 1./sqrt(2)
    gen = [
        [[0, 1], [1, 0]],  # X
        [[1, 0], [0, -1]], # Z
        [[r, r], [r, -r]], # Hadamard
    ]
    gen = [Qu((2, 2), 'ud', v) for v in gen]

    RealCliff = mulclose(gen) # 
    assert len(RealCliff)==16 # D_16 ?

    # ----------------------------------
    #

    gen = [
        [[0, 1], [1, 0]],  # X
        [[1, 0], [0, -1]], # Z
        [[1, 0], [0, 1.j]], # S
        [[r, r], [r, -r]], # Hadamard
    ]
    gen = [Qu((2, 2), 'ud', v) for v in gen]

    Cliff = mulclose(gen) # Is this the correct name?
    assert len(Cliff)==192

    # ----------------------------------
    #

    gen = [
        [[0, 1], [1, 0]],
        [[1, 0], [0, -1]],
        [[0, 1.j], [-1.j, 0]],
    ]
    gen = [Qu((2, 2), 'ud', v) for v in gen]

    Pauli = mulclose(gen)
    assert len(Pauli)==16

    # ----------------------------------
    # binary octahedral group

    x = cyclotomic(8)

    a = x-x**3 # sqrt(2)
    i = x**2

    gen = [
        [[(-1+i)/2,(-1+i)/2], [(1+i)/2,(-1-i)/2]], 
        [[(1+i)/a,0], [0,(1-i)/a]]
    ]
    gen = [Qu((2, 2), 'ud', v) for v in gen]

    Octa = mulclose(gen)
    assert len(Octa)==48

    # ----------------------------------
    # binary tetrahedral group ... hacked

    i = cyclotomic(4)

    gen = [
        [[(-1+i)/2, (-1+i)/2], [(1+i)/2, (-1-i)/2]],
        [[0,i], [-i,0]]
    ]

    gen = [Qu((2, 2), 'ud', v) for v in gen]

    Tetra = mulclose(gen)
    assert len(Tetra)==48 # whoops it must be another Octa ... 
    
    Tetra = [g for g in Tetra if g in Octa]  # hack this
    Tetra = mulclose(Tetra)
    assert len(Tetra)==24 # works!

    # ----------------------------------
    # binary icosahedral group

    v = cyclotomic(10)
    z5 = v**2
    i = z5**5
    a = 2*z5**3 + 2*z5**2 + 1 #sqrt(5)
    gen = [
        [[z5**3,0], [ 0,z5**2]], 
        [[0,1], [ -1,0]], 
        [[(z5**4-z5)/a, (z5**2-z5**3)/a], [ (z5**2-z5**3)/a, -(z5**4-z5)/a]]
    ]
    
    gen = [Qu((2, 2), 'ud', v) for v in gen]

    Icosa = mulclose(gen)
    assert len(Icosa)==120


build()


#class Ring(object):
#    def __init__(self):


def test_c():

    "build invariant commutative polynomials"

    I = Poly.identity(2)
    zero = Poly.zero(2)
    x = Poly({(1, 0): 1.})
    y = Poly({(0, 1): 1.})

    #print( (x+I) ** 3 )

    def act(g, p):
        #print("act", g, p)
        #print(g[0, 0])
        x1 = g[0, 0]*x + g[1, 0]*y
        y1 = g[0, 1]*x + g[1, 1]*y
        s = str(p)
        p = eval(s, {}, {"x":x1, "y":y1})
        return p

    def orbit_sum(G, p):
        p1 = zero
        for g in G:
            p1 = p1 + act(g, p)
        return p1

    assert orbit_sum(Sym, x*y**2) == x*y**2 + x**2*y

    #p = orbit_sum(Tetra, x**8)
    #print(p)

    G = eval(argv.get("G", "Tetra"))

    degree = argv.get("degree", 8)
    for degree in range(1, degree+1):
        for i in range(degree+1):
            items = [x]*i + [y]*(degree-i)
            assert len(items)==degree
            p = reduce(mul, items)
            q = orbit_sum(G, p)
            if q.degree > 0:
                print("degree:", q.degree)
                print(q)


class Tensor(object):

    """ Some kind of graded ring element... I*I*I + X*X*X etc.
        There is no real reason to make this homogeneous,
        but i do for now.
    """

    zero = 0.0
    one = 1.0
    def __init__(self, items, grade=None):
        # map key -> coeff, key is ("A", "B") etc.
        assert items or (grade is not None)
        keys = list(items.keys())
        keys.sort()
        self.items = {} 
        nz = [] 
        for key in keys:
            assert grade is None or grade==len(key)
            grade = len(key)
            v = items[key]
            if abs(v) > EPSILON:
                self.items[key] = v # uniquify
                nz.append(key)
        self.keys = nz 
        self.grade = grade

    def get_zero(self):
        return Tensor({}, self.grade)

    def __add__(self, other):
        assert self.grade == other.grade # i guess this is not necessary...
        items = dict(self.items)
        for (k, v) in other.items.items():
            items[k] = items.get(k, self.zero) + v
        return Tensor(items, self.grade)

    def __sub__(self, other):
        assert self.grade == other.grade
        items = dict(self.items)
        for (k, v) in other.items.items():
            items[k] = items.get(k, self.zero) - v
        return Tensor(items, self.grade)

    def __matmul__(self, other):
        items = {} 
        for (k1, v1) in self.items.items():
          for (k2, v2) in other.items.items():
            k = k1+k2
            assert k not in items
            items[k] = v1*v2
        return Tensor(items, self.grade+other.grade)

    def __rmul__(self, r):
        items = {}
        for (k, v) in self.items.items():
            items[k] = r*v
        return Tensor(items, self.grade)

    def __len__(self):
        return len(self.items)

    def subs(self, rename):
        the_op = Tensor({}, self.grade) # zero
        one = self.one
        for (k, v) in self.items.items():
            final = None
            for ki in k:
                op = rename.get(ki, Tensor({ki : one}))
                if final is None:
                    final = op
                else:
                    final = final @ op # tensor
            the_op = the_op + v*final
        return the_op

    def evaluate(self, rename):
        the_op = None
        one = self.one
        for (k, v) in self.items.items():
            final = None
            for ki in k:
                op = rename[ki]
                if final is None:
                    final = op
                else:
                    final = final @ op # tensor
            the_op = v*final if the_op is None else the_op + v*final
        return the_op

    def __str__(self):
        ss = []
        for k in self.keys:
            v = self.items[k]
            s = ''.join(str(ki) for ki in k)
            if abs(v-1) < EPSILON:
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
        return "Tensor(%s)"%(self.items)

    def norm(self):
        return sum(abs(val) for val in self.items.values())

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


def test_nc():

    "build invariant non-commutative polynomials"

    I = Tensor({"I" : 1})
    X = Tensor({"X" : 1})
    Y = Tensor({"Y" : 1})
    Z = Tensor({"Z" : 1})

    II = I@I
    XI = X@I
    IX = I@X
    XX = X@X
    assert II+II == 2*II

    assert X@(XI + IX) == X@X@I + X@I@X

    assert ((I-Y)@I + I@(I-Y)) == 2*I@I - I@Y - Y@I
    assert (XI + IX).subs({"X": I-Y}) == ((I-Y)@I + I@(I-Y))

    A = Tensor({"A":1})
    B = Tensor({"B":1})
    p = A@A@A + B@B@A + B@A@B + A@B@B
    q = A@A@A + B@B@B
    p1 = p.subs({"A": A+B, "B": A-B})
    assert p1 == 4*A@A@A + 4*B@B@B


    def act(g, op):
        A1 = g[0, 0]*A + g[1, 0]*B
        B1 = g[0, 1]*A + g[1, 1]*B
        op = op.subs({"A" : A1, "B" : B1})
        return op

    def orbit_sum(G, p):
        p1 = p.get_zero()
        for g in G:
            p1 = p1 + act(g, p)
        return p1

    assert orbit_sum(Sym, A@A) == A@A + B@B


    G = eval(argv.get("G", "Tetra"))
    degree = argv.get("degree", 8)
    #items = [A]*degree
    for i in range(degree+1):
        items = [A]*i + [B]*(degree-i)
        assert len(items)==degree
        p = reduce(matmul, items)
        p1 = orbit_sum(G, p)
        if p1==0:
            continue
        if argv.show:
            print(p1)
            print("terms:", len(p1))

    return

    if argv.check:
        for g in G:
            assert act(g, p1) == p1
            print(".", end=" ", flush=True)

    I = Qu((2, 2), 'ud', [[1, 0], [0, 1]])
    X = Qu((2, 2), 'ud', [[0, 1], [1, 0]])
    Z = Qu((2, 2), 'ud', [[1, 0], [0, -1]])
    Y = Qu((2, 2), 'ud', [[0, 1.j], [-1.j, 0]])
    
    P = p1.evaluate({"A" : I, "B" : X})
    print(P.shape)
    #P /= len(G)

    print("building transversal gates")
    GT = []
    count = 0
    for g in G:
        op = reduce(matmul, [g]*degree)
        #GT.append(op)
        if op*P == P*op:
            count += 1
            print(".", end=" ", flush=True)
    print("count:", count)
    return

    items = P.eigs()
    for val, vec in items:
        print(fstr(val), end=" ")

    for val, vec in items:
        #print(val, vec.shortstr())
        #if abs(val) < EPSILON:
        #    continue
        #print()
        #print("-"*79)
        print("val=%s" % fstr(val), end=" ")

        vec.normalize()
        count = 0
        for g in GT:
            wec = g*vec
            if (wec-vec).norm() < 1e-6:
                count += 1
            #r = ~wec*vec
            #print(fstr(r), end=" ")
        print("stab:", count, end=" ", flush=True)

#        for g in GT:
#            wec = g*vec
#            xec = P*wec
#            if xec.norm()<EPSILON:
#                continue
#            r = ~(xec.normalized())*wec
#            print(fstr(r), end=" ")

    print()



    



if __name__ == "__main__":

    name = argv.next() or "main"

    fn = eval(name)
    fn()


