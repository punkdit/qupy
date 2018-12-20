#!/usr/bin/env python3

"""
Fooling around with polynomial invariants of
finite subgroups of SU(2), both commutative and
non-commutative polynomials.
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
from qupy.util import mulclose, mulclose_names, show_spec
from qupy.tool import fstr, astr, cross, write
from qupy.argv import argv

from qupy.dev.comm import Poly

EPSILON=1e-8


def cyclotomic(n):
    return numpy.exp(2*numpy.pi*1.j/n)


def build():

    global Octa, Tetra, Icosa, Sym2, Sym3, Pauli, RealPauli, RealCliff, Cliff

    # ----------------------------------
    #
    
    gen = [
        [[0, 1], [1, 0]],
    ]
    gen = [Qu((2, 2), 'ud', v) for v in gen]

    Sym2 = mulclose(gen)
    assert len(Sym2)==2


    # ----------------------------------
    #

    gen = [
        [[-1, 1], [0, 1]],
        [[1, 0], [1, -1]],
    ]
    gen = [Qu((2, 2), 'ud', v) for v in gen]

    Sym3, words = mulclose_names(gen, "ab")
    assert len(Sym3)==6

    # ----------------------------------
    #

    gen = [
        [[0, 1], [1, 0]],
        [[1, 0], [0, -1]],
    ]
    gen = [Qu((2, 2), 'ud', v) for v in gen]

    RealPauli, words = mulclose_names(gen, 'XZ')
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

    RealCliff, words = mulclose_names(gen, "XZH") # 
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
    cliff_gen = gen

    Cliff, words = mulclose_names(gen, "XZSH") # Is this the correct name?
    assert len(Cliff)==192

    # ----------------------------------
    #

    gen = [
        [[0, 1], [1, 0]],
        [[1, 0], [0, -1]],
        [[0, 1.j], [-1.j, 0]],
    ]
    gen = [Qu((2, 2), 'ud', v) for v in gen]

    Pauli, words = mulclose_names(gen, "XZY")
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
    octa_gen = gen

    Octa, words = mulclose_names(gen, "AB")
    assert len(Octa)==48
    print("Octa:", words.values())

    # Octa is a subgroup of Cliff:
    for g in octa_gen:
        assert g in Cliff

    # ----------------------------------
    # binary tetrahedral group ... hacked

    i = cyclotomic(4)

    gen = [
        [[(-1+i)/2, (-1+i)/2], [(1+i)/2, (-1-i)/2]],
        [[0,i], [-i,0]]
    ]

    gen = [Qu((2, 2), 'ud', v) for v in gen]

    Tetra, words = mulclose_names(gen, "CD")
    assert len(Tetra)==48 # whoops it must be another Octa ... 
    
    Tetra = [g for g in Tetra if g in Octa]  # hack this
    Tetra, words = mulclose_names(Tetra, [words[g] for g in Tetra])
    #print("Tetra:", words.values())
    assert len(Tetra)==24 # _works!

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

    Icosa, words = mulclose_names(gen, "EFG")
    #print("Icosa:", words.values())
    assert len(Icosa)==120


build()


#class Ring(object):
#    def __init__(self):

def bravyi_kitaev():
    "mysterious T gate from 2005 PRA paper"

    I = Qu((2, 2), 'ud', [[1., 0.], [0., 1.]])
    T = (1./sqrt(2)) * exp(1.j*pi/4) * Qu((2, 2), 'ud',
        [[1., 1.], [1.j, -1.j]])

    assert T != I
    assert T**2 != I
    assert T**3 == -I

    assert T in Octa
    


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

    assert orbit_sum(Sym2, x*y**2) == x*y**2 + x**2*y

    #p = orbit_sum(Tetra, x**8)
    #print(p)

    G = eval(argv.get("G", "Tetra"))

    dims = []
    degree = argv.get("degree", 8)
    for degree in range(1, degree+1):
        ops = []
        for i in range(degree+1):
            items = [x]*i + [y]*(degree-i)
            assert len(items)==degree
            p = reduce(mul, items)
            q = orbit_sum(G, p)
            if q.degree > 0:
                #print("degree:", q.degree)
                #print(q)
                ops.append(q)
        if not ops:
            dims.append(0)
            continue
        #print("degree:", degree)
        #print("found:", len(ops))
        _ops = linear_independent(ops)
        if argv.show:
            for op in _ops:
                print(op)
        #print("dimension:", len(_ops))
        print("[degree=%d dim=%d]" % (degree, len(_ops)))
        dims.append(len(_ops))
    print(",".join(str(dim) for dim in dims))


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

    def get_keys(self):
        return list(self.keys)

    def __getitem__(self, key):
        return self.items.get(key, self.zero)

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


def swap_row(A, j, k):
    row = A[j, :].copy()
    A[j, :] = A[k, :]
    A[k, :] = row


def swap_col(A, j, k):
    col = A[:, j].copy()
    A[:, j] = A[:, k]
    A[:, k] = col


def row_reduce(A, truncate=False, inplace=False, check=False, verbose=False):
    """ Remove zero rows if truncate==True
    """

    assert len(A.shape)==2, A.shape
    m, n = A.shape
    if not inplace:
        A = A.copy()

    if m*n==0:
        if truncate and m:
            A = A[:0, :]
        return A

    if verbose:
        print("row_reduce")
        #print("%d rows, %d cols" % (m, n))

    i = 0
    j = 0
    while i < m and j < n:
        if verbose:
            print("i, j = %d, %d" % (i, j))
            print("A:")
            print(shortstrx(A))

        assert i<=j
        if i and check:
            assert (numpy.abs(A[i:,:j])>EPSILON).sum() == 0

        # first find a nonzero entry in this col
        for i1 in range(i, m):
            if abs(A[i1, j])>EPSILON:
                break
        else:
            j += 1 # move to the next col
            continue # <----------- continue ------------

        if i != i1:
            if verbose:
                print("swap", i, i1)
            swap_row(A, i, i1)

        assert abs(A[i, j]) > EPSILON
        for i1 in range(i+1, m):
            if abs(A[i1, j])>EPSILON:
                if verbose:
                    print("add row %s to %s" % (i, i1))
                r = -A[i1, j] / A[i, j]
                A[i1, :] += r*A[i, :]
                assert abs(A[i1, j]) < EPSILON

        i += 1
        j += 1

    if truncate:
        m = A.shape[0]-1
        #print("sum:", m, A[m, :], A[m, :].sum())
        while m>=0 and (numpy.abs(A[m, :])>EPSILON).sum()==0:
            m -= 1
        A = A[:m+1, :]

    if verbose:
        print()

    return A


def linear_independent(ops):
    "ops: list of Tensor's or list of Poly's"
    assert len(ops) 
    keys = set()
    for op in ops:
        keys.update(op.get_keys())
        cls = op.__class__ # Poly or Tensor
    keys = list(keys)
    keys.sort()
    #print("keys:", keys)

    #A = numpy.zeros((len(ops), len(keys)), dtype=numpy.complex128)
    A = numpy.zeros((len(ops), len(keys)))
    for i, op in enumerate(ops):
        for j, key in enumerate(keys):
            val = op[key]
            A[i, j] = val.real

    #print("A:")
    #print(A)
    B = row_reduce(A, truncate=True)
    #print("B:")
    #print(B)

    m = len(B)
    _ops = []
    for i in range(m):
        cs = {}
        for j, key in enumerate(keys):
            if abs(B[i, j])>EPSILON:
                cs[key] = B[i, j]
        assert len(cs)
        op = cls(cs)
        _ops.append(op)
    return _ops


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
        # argh,  numpy.complex128 doesn't play nice with Tensor.__getitem__
        A1 = complex(g[0, 0])*A + complex(g[1, 0])*B 
        B1 = complex(g[0, 1])*A + complex(g[1, 1])*B
        op = op.subs({"A" : A1, "B" : B1})
        return op

    def orbit_sum(G, p):
        p1 = p.get_zero()
        for g in G:
            p1 = p1 + act(g, p)
        return p1

    assert orbit_sum(Sym2, A@A) == A@A + B@B

    def show_spec(P):
        items = P.eigs()
        for val, vec in items:
            print(fstr(val), end=" ")
        print()

    I = Qu((2, 2), 'ud', [[1, 0], [0, 1]])
    X = Qu((2, 2), 'ud', [[0, 1], [1, 0]])
    Z = Qu((2, 2), 'ud', [[1, 0], [0, -1]])
    Y = Qu((2, 2), 'ud', [[0, 1.j], [-1.j, 0]])

    G = eval(argv.get("G", "Tetra"))
    degree = argv.get("degree", 8)
    ops = []
    pair = [A, B]
    remain = [items for items in cross((pair,)*degree)]
    while remain:
        items = iter(remain).__next__()
        remain.remove(items)
        assert len(items)==degree
        p = reduce(matmul, items)
        p1 = orbit_sum(G, p)
        if p1==0:
            continue
        ops.append(p1)
        print(".", end=" ", flush=True)
    print()

    print("found ops:", len(ops))
    if not ops:
        return
    _ops = linear_independent(ops)
    print("dimension:", len(_ops))

    if argv.show:
        for op in _ops:
            print(op)

    return

    for p1 in ops:
    
        P = p1.evaluate({"A" : I, "B" : X})
        #print(P.shape)
        #P /= len(G)
    
        print("transversal gates...")
        GT = []
        count = 0
        for g in G:
            op = reduce(matmul, [g]*degree)
            #GT.append(op)
            if op*P == P*op:
                count += 1
                print(".", end=" ", flush=True)
        print("commutes with", count, "transversal gates")
        print("spec:", end=" ")
        show_spec(P)

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



def build_quaternion():

    # See:
    # https://homepages.warwick.ac.uk/~masda/McKay/Carrasco_Project.pdf

    global Q8, QT, QO, QI, Sym44, Sym54

    def quaternion(a, b, c, d): 
        # build matrix representation of quaternion
        A = numpy.array([
            [a, -b, -c, -d],
            [b, a, -d, c], 
            [c, d, a, -b],
            [d, -c, b, a]])
        A = Qu((4, 4), 'ud', A)
        return A
    
    e = quaternion(1, 0, 0, 0)
    i = quaternion(0, 1, 0, 0)
    j = quaternion(0, 0, 1, 0)
    k = quaternion(0, 0, 0, 1)

    # ----------------------------------
    #

    gen = [i, j, k]
    Q8 = mulclose(gen)
    assert len(Q8)==8

    # ----------------------------------
    #

    QT = mulclose([i, j, 0.5*(e+i-j+k)])
    assert len(QT)==24

    # ----------------------------------
    #

    QO = mulclose([
        (e+i)/sqrt(2), j, (e+i-j+k)/2.])
    assert len(QO)==48

    # ----------------------------------
    #

    z = (1+sqrt(5))/2.
    QI = mulclose([
        j, (e+i+j+k)/2., (z*e + (1./z)*i + j)/2.], maxsize=200)
    assert len(QI)==120

    # ----------------------------------
    #

    s1 = numpy.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    
    s2 = numpy.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]])
    
    s3 = numpy.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]])

    gen = [Qu((4, 4), 'ud', g) for g in [s1, s2, s3]]
    Sym44 = mulclose(gen)
    
    # ----------------------------------
    #

    gen = [
        [[-1, 1], [0, 1]],
        [[1, 0], [1, -1]],
    ]
    s1 = numpy.array([
        [-1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    
    s2 = numpy.array([
        [1, 0, 0, 0],
        [1, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    
    s3 = numpy.array([
        [1, 0, 0, 0],
        [0, -1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

    # gap> IrreducibleRepresentations(SymmetricGroup(5));
    gen = [
        [[ 0, -1, 0, 0 ], [ 0, 0, 0, 1 ], [ 1, -1, -1, -1 ], [ 0, 0, 1, 0 ]], 
        [[ -1, 0, 0, 0 ], [ 1, -1, -1, -1 ], [ -1, 0, 0, 1 ], [ -1, 0, 1, 0 ]]] 

    gen = [Qu((4, 4), 'ud', v) for v in gen]

    Sym54 = mulclose(gen)
    assert len(Sym54)==24*5


build_quaternion()


def test_code():

    I = Tensor({"I" : 1})
    X = Tensor({"X" : 1})
    Z = Tensor({"Z" : 1})
    Y = Tensor({"Y" : 1})

    II = I@I
    XI = X@I
    IX = I@X
    XX = X@X
    assert II+II == 2*II

    assert X@(XI + IX) == X@X@I + X@I@X

    assert ((I-Y)@I + I@(I-Y)) == 2*I@I - I@Y - Y@I
    assert (XI + IX).subs({"X": I-Y}) == ((I-Y)@I + I@(I-Y))

    if argv.five:
        op = (I@I@I@I@I+I@X@Z@Z@X-I@Z@Y@Y@Z-I@Y@X@X@Y
            +X@I@X@Z@Z-X@X@Y@I@Y+X@Z@Z@X@I-X@Y@I@Y@X
            -Z@I@Z@Y@Y+Z@X@I@X@Z+Z@Z@X@I@X-Z@Y@Y@Z@I
            -Y@I@Y@X@X-Y@X@X@Y@I-Y@Z@I@Z@Y-Y@Y@Z@I@Z)
    elif argv.steane:
        op = (I@I@I@I@I@I@I+I@I@X@X@X@X@I+I@I@Z@Z@Z@Z@I+I@I@Y@Y@Y@Y@I
            +I@X@I@X@X@I@X+I@X@X@I@I@X@X+I@X@Z@Y@Y@Z@X+I@X@Y@Z@Z@Y@X
            +I@Z@I@Z@Z@I@Z+I@Z@X@Y@Y@X@Z+I@Z@Z@I@I@Z@Z+I@Z@Y@X@X@Y@Z
            +I@Y@I@Y@Y@I@Y+I@Y@X@Z@Z@X@Y+I@Y@Z@X@X@Z@Y+I@Y@Y@I@I@Y@Y
            +X@I@I@X@I@X@X+X@I@X@I@X@I@X+X@I@Z@Y@Z@Y@X+X@I@Y@Z@Y@Z@X
            +X@X@I@I@X@X@I+X@X@X@X@I@I@I+X@X@Z@Z@Y@Y@I+X@X@Y@Y@Z@Z@I
            +X@Z@I@Y@Z@X@Y+X@Z@X@Z@Y@I@Y+X@Z@Z@X@I@Y@Y+X@Z@Y@I@X@Z@Y
            +X@Y@I@Z@Y@X@Z+X@Y@X@Y@Z@I@Z+X@Y@Z@I@X@Y@Z+X@Y@Y@X@I@Z@Z
            +Z@I@I@Z@I@Z@Z+Z@I@X@Y@X@Y@Z+Z@I@Z@I@Z@I@Z+Z@I@Y@X@Y@X@Z
            +Z@X@I@Y@X@Z@Y+Z@X@X@Z@I@Y@Y+Z@X@Z@X@Y@I@Y+Z@X@Y@I@Z@X@Y
            +Z@Z@I@I@Z@Z@I+Z@Z@X@X@Y@Y@I+Z@Z@Z@Z@I@I@I+Z@Z@Y@Y@X@X@I
            +Z@Y@I@X@Y@Z@X+Z@Y@X@I@Z@Y@X+Z@Y@Z@Y@X@I@X+Z@Y@Y@Z@I@X@X
            +Y@I@I@Y@I@Y@Y+Y@I@X@Z@X@Z@Y+Y@I@Z@X@Z@X@Y+Y@I@Y@I@Y@I@Y
            +Y@X@I@Z@X@Y@Z+Y@X@X@Y@I@Z@Z+Y@X@Z@I@Y@X@Z+Y@X@Y@X@Z@I@Z
            +Y@Z@I@X@Z@Y@X+Y@Z@X@I@Y@Z@X+Y@Z@Z@Y@I@X@X+Y@Z@Y@Z@X@I@X
            +Y@Y@I@I@Y@Y@I+Y@Y@X@X@Z@Z@I+Y@Y@Z@Z@X@X@I+Y@Y@Y@Y@I@I@I)
    elif argv.rand:
        op = I@I@I@I-I@Y@Y@X-X@I@Z@Z+X@Y@X@Y+Z@I@X@Z+Z@Y@Z@Y+Y@I@Y@I+Y@Y@I@X
    else:
        op = argv.op
        if op:
            op = eval(op, locals())

    if op is not None:
        ops = [op]
        print(op)
    
    def act(g, op):
        # argh,  numpy.complex128 doesn't play nice with Tensor.__getitem__
        I1 = complex(g[0, 0])*I+complex(g[1, 0])*X+complex(g[2, 0])*Z+complex(g[3, 0])*Y 
        X1 = complex(g[0, 1])*I+complex(g[1, 1])*X+complex(g[2, 1])*Z+complex(g[3, 1])*Y 
        Z1 = complex(g[0, 2])*I+complex(g[1, 2])*X+complex(g[2, 2])*Z+complex(g[3, 2])*Y 
        Y1 = complex(g[0, 3])*I+complex(g[1, 3])*X+complex(g[2, 3])*Z+complex(g[3, 3])*Y 
        op = op.subs({"I":I1, "X":X1, "Z":Z1, "Y":Y1})
        return op

    def orbit_sum(G, op):
        p1 = op.get_zero()
        for g in G:
            p1 = p1 + act(g, op)
        return p1

    if op is None:
        G = eval(argv.get("G4", "QT")) # the "external" group
        print("|G|=", len(G))
    
        #p1 = orbit_sum(G, op)
        #print(p1)
    
        degree = argv.get("degree", 2)
        ops = []
        pair = [I, X, Z, Y]
        remain = [items for items in cross((pair,)*degree)]
        print("remain:", len(remain))
        while remain:
            items = iter(remain).__next__()
            remain.remove(items)
            assert len(items)==degree
            p = reduce(matmul, items)
            p1 = orbit_sum(G, p)
            if p1==0:
                continue
            ops.append(p1)
            #print(".", end=" ", flush=True)
        #print()
    
        print("found ops:", len(ops))
        if not ops:
            return
        ops = linear_independent(ops)
        print("dimension:", len(ops))

    if argv.show:
        for op in ops:
            print(op)

    # ------------------------------------------

    if not argv.find_transversal:
        return

    def show_spec(P):
        items = P.eigs()
        for val, vec in items:
            print(fstr(val), end=" ")
        print()

    G = eval(argv.get("G2", "Tetra")) # the "internal" group

    def find_transversal(op):
        I = Qu((2, 2), 'ud', [[1, 0], [0, 1]])
        X = Qu((2, 2), 'ud', [[0, 1], [1, 0]])
        Z = Qu((2, 2), 'ud', [[1, 0], [0, -1]])
        #Y = Qu((2, 2), 'ud', [[0, 1.j], [-1.j, 0]])
        Y = X*Z
        
        P = op.evaluate({"I":I, "X":X, "Z":Z, "Y":Y})
        #print(P.shape)
        #P /= len(G)
    
        degree = op.grade
    
        GT = []
        count = 0
        for g in G:
            op = reduce(matmul, [g]*degree)
            #GT.append(op)
            if op*P == P*op:
                count += 1
                print(".", end=" ", flush=True)
        print("commutes with", count, "transversal gates")
        if argv.show_spec:
            print("spec:")
            show_spec(P)

    print("transversal gates...")
    for op in ops:
        find_transversal(op)

def main():
    pass



if __name__ == "__main__":

    name = argv.next() or "main"

    fn = eval(name)
    fn()


