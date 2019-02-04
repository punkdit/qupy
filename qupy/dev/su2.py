#!/usr/bin/env python3

"""
Fooling around with polynomial invariants of
finite subgroups of SU(2), both commutative and
non-commutative polynomials.
"""

from math import sqrt
from random import choice, seed

from functools import reduce
from operator import mul, matmul, add

import numpy
from numpy import exp, pi

#from qupy import scalar
#scalar.use_reals()
#from qupy.scalar import EPSILON, MAX_GATE_SIZE

from qupy.dense import Qu
from qupy.util import mulclose, mulclose_names, show_spec
from qupy.tool import fstr, astr, cross, write
from qupy.argv import argv
from qupy.dev.comm import Poly
from qupy.dev import groups
from qupy.dev.linalg import row_reduce
from qupy.dev.assoc import AssocAlg


if argv.fast:
    print("importing _algebra")
    from qupy.dev._algebra import Algebra, build_algebra, Tensor
else:
    from qupy.dev.algebra import Algebra, build_algebra, Tensor


EPSILON=1e-8


def sumn(items):
    items = list(items)
    op = items[0]
    for opi in items[1:]:
        op = op + opi
    return op


def cyclotomic(n):
    return numpy.exp(2*numpy.pi*1.j/n)



class Group(object):
    def __init__(self, gen, names=None, **kw):
        if names is None:
            names = "ABCDEFGHIJKLMNPQRSTUVWXYZ"[:len(gen)]
        assert len(gen)==len(names)
        G, words = mulclose_names(gen, names, **kw)
        self.G = G
        self.words = words
        self.gen = gen
        self.identity = None
        self.inv = None

    def build(self):
        if self.identity is not None:
            return
        G = self.G
        n = len(G)
    
        # group identity
        identity = None
        for i in range(n):
            for j in range(i+1, n):
                g = G[i]*G[j]
                k = G.index(g)
                if k==i:
                    assert identity is None or identity==j
                    identity = j
                    break
    
        # group _inverse
        inv = [None]*len(G)
        for i, g in enumerate(G):
            if inv[i] is not None:
                continue
            for j, h in enumerate(G):
                gh = g*h
                k = G.index(gh)
                if k==identity:
                    assert inv[i] is None or inv[i]==j
                    inv[i] = j
                    inv[j] = i
                    break

        self.identity = identity
        self.inv = inv

    def index(self, g):
        return self.G.index(g)

    def __getitem__(self, idx):
        return self.G[idx]

    def __len__(self):
        return len(self.G)



def build():

    global Octa, Tetra, Icosa, Sym2, Sym3, Pauli, RealPauli, RealCliff, Cliff

    # ----------------------------------
    #
    
    gen = [
        [[0, 1], [1, 0]],
    ]
    gen = [Qu((2, 2), 'ud', v) for v in gen]

    Sym2 = Group(gen, "a")
    assert len(Sym2)==2


    # ----------------------------------
    #

    gen = [
        [[-1, 1], [0, 1]],
        [[1, 0], [1, -1]],
    ]
    gen = [Qu((2, 2), 'ud', v) for v in gen]

    Sym3 = Group(gen, "ab")
    assert len(Sym3)==6

    # ----------------------------------
    #

    gen = [
        [[0, 1], [1, 0]],
        [[1, 0], [0, -1]],
    ]
    gen = [Qu((2, 2), 'ud', v) for v in gen]

    RealPauli = Group(gen, 'XZ')
    assert len(RealPauli)==8

    # ----------------------------------
    #

    r = 1./sqrt(2)
    gen = [
        [[1, 0], [0, -1]], # Z
        [[r, r], [r, -r]], # Hadamard
    ]
    gen = [Qu((2, 2), 'ud', v) for v in gen]

    RealCliff = Group(gen, "ZH") # 
    assert len(RealCliff)==16 # D_16

    # ----------------------------------
    #

    gen = [
        [[0, 1], [1, 0]],  # X == HZH
        [[1, 0], [0, -1]], # Z == S*S
        [[1, 0], [0, 1.j]], # S
        [[r, r], [r, -r]], # Hadamard
    ]
    gen = [Qu((2, 2), 'ud', v) for v in gen]
    cliff_gen = gen

    X, Z, S, H = gen
    assert Z == S*S
    assert X == H*Z*H

    Cliff = Group(gen, "XZSH") # Is this the correct name?
    assert len(Cliff)==192

    # ----------------------------------
    #

    gen = [
        [[0, 1], [1, 0]],
        [[1, 0], [0, -1]],
        [[0, 1.j], [-1.j, 0]],
    ]
    gen = [Qu((2, 2), 'ud', v) for v in gen]

    Pauli = Group(gen, "XZY")
    assert len(Pauli)==16

    # ----------------------------------
    # binary octahedral group

    idx = argv.get("idx", 0)

    gen, G = groups.build_octa(idx)
    octa_gen = gen

    Octa = Group(gen, "AB")
    assert len(Octa)==48
    #print("Octa:", Octa.words.values())

    # ----------------------------------
    # binary tetrahedral group 

    gen, G = groups.build_tetra(idx)

    Tetra = Group(gen, "CD")
    assert len(Tetra)==24

    global TetraXZ
    gen = gen + [X, Z]
    TetraXZ = Group(gen, "CDXZ")
    assert len(TetraXZ) == 48

    #for g in TetraXZ:
    #    assert g in Octa # nope...

    # ----------------------------------
    # binary icosahedral group

    gen, G = groups.build_icosa(idx)

    Icosa = Group(gen, "EF")
    #print("Icosa:", words.values())
    assert len(Icosa)==120

    # ----------------------------------

    if 0:
        for g in Tetra:
            assert g in Octa
    
        # Octa is a subgroup of Cliff:
        for g in octa_gen:
            assert g in Cliff



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

    assert T in Tetra


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



def linear_independent(ops, construct=None, epsilon=EPSILON):
    "ops: list of Tensor's or list of Poly's"
    assert len(ops) 
    keys = set()
    for op in ops:
        keys.update(op.get_keys())
        if construct is None:
            construct = op.__class__ # Poly or Tensor
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
    B = row_reduce(A, truncate=True, epsilon=epsilon)
    #print("B:")
    #print(B)

    m = len(B)
    _ops = []
    for i in range(m):
        cs = {}
        for j, key in enumerate(keys):
            if abs(B[i, j])>epsilon:
                cs[key] = B[i, j]
        assert len(cs)
        op = construct(cs)
        _ops.append(op)
    return _ops


def test_pauli():
    #algebra = Algebra("IXYZ")
    pauli = build_algebra("IXZY",
        "X*X=I Z*Z=I Y*Y=-I X*Z=Y Z*X=-Y X*Y=Z Y*X=-Z Z*Y=-X Y*Z=X")

    I = pauli.I
    X = pauli.X
    Y = pauli.Y
    Z = pauli.Z

    assert I*X == X
    assert X*X==I 
    assert Z*Z==I 
    assert Y*Y==-I 
    assert X*Z==Y 
    assert Z*X==-Y 
    assert X*Y==Z 
    assert Y*X==-Z 
    assert Z*Y==-X 
    assert Y*Z==X

    II = I@I
    XI = X@I
    IX = I@X
    XX = X@X
    assert II+II == 2*II

    assert X@(XI + IX) == X@X@I + X@I@X

    assert ((I-Y)@I + I@(I-Y)) == 2*I@I - I@Y - Y@I
    assert (XI + IX).subs({"X": I-Y}) == ((I-Y)@I + I@(I-Y))


def test_nc():

    "build invariant non-commutative polynomials with two variables"

    algebra = Algebra(2, "AB") # no multiplication specified
    A = algebra.A
    B = algebra.B
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



def build_external():

    # See:
    # https://homepages.warwick.ac.uk/~masda/McKay/Carrasco_Project.pdf

    global Q8, QT, QO, QI, Sym44, Sym54, H4

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
    Q8 = Group(gen)
    assert len(Q8)==8

    # ----------------------------------
    #

    QT = Group([i, j, 0.5*(e+i-j+k)])
    assert len(QT)==24

    # ----------------------------------
    #

    QO = Group([
        (e+i)/sqrt(2), j, (e+i-j+k)/2.])
    assert len(QO)==48

    # ----------------------------------
    #

    z = (1+sqrt(5))/2.
    QI = Group([
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
    Sym44 = Group(gen)
    
    # ----------------------------------
    #

    # gap> IrreducibleRepresentations(SymmetricGroup(5));
    gen = [
        [[ 0, -1, 0, 0 ], [ 0, 0, 0, 1 ], [ 1, -1, -1, -1 ], [ 0, 0, 1, 0 ]], 
        [[ -1, 0, 0, 0 ], [ 1, -1, -1, -1 ], [ -1, 0, 0, 1 ], [ -1, 0, 1, 0 ]]] 

    gen = [Qu((4, 4), 'ud', v) for v in gen]

    Sym54 = Group(gen)
    assert len(Sym54)==24*5

    # ----------------------------------
    #

    H = 1./2*numpy.array([
        [1, 1, 1, 1],
        [1, 1, -1, -1],
        [1, -1, 1, -1],
        [1, -1, -1, 1]])

    H = Qu((4, 4), 'ud', H)

    H4 = Group([H])
    assert len(H4) == 2


build_external()


def decompose(basis, op):
    _basis = []
    norms = []
    for a in basis: 
        a = a.v.flatten()
        r = numpy.linalg.norm(a)
        a = a/r
        norms.append(r)
        _basis.append(a)
    norms = numpy.array(norms)
    U = numpy.array(_basis)
    Ut = U.transpose()
    I = numpy.dot(Ut, U)
    assert numpy.allclose(I, numpy.eye(len(basis))), "need orthonormal basis"
    v = op.v.flatten()
    #r = numpy.linalg.norm(v)
    v = numpy.dot(U, v)
    v = v/r # hack this
    return v


def promote_pauli(pauli, G):
    I = Qu((2, 2), 'ud', [[1, 0], [0, 1]])
    X = Qu((2, 2), 'ud', [[0, 1], [1, 0]])
    Z = Qu((2, 2), 'ud', [[1, 0], [0, -1]])
    Y = X*Z
        
    basis = [I, X, Z, Y]
    v = decompose(basis, I)
    assert numpy.allclose(v, [1, 0, 0, 0])

    v = decompose(basis, I+X)
    assert numpy.allclose(v, [1, 1, 0, 0])

    #pauli = build_algebra("IXZY",
    #    "X*X=I Z*Z=I Y*Y=-I X*Z=Y Z*X=-Y X*Y=Z Y*X=-Z Z*Y=-X Y*Z=X")

    for op in G:
        v = decompose(basis, op)
        #g = Tensor({}, 1, pauli)
        g = pauli.get_zero(1)
        for i in range(pauli.dim):
            g = g + complex(v[i])*pauli.basis[i] # ARGH!
        yield g


def show_algebra():
    A = Qu((2, 2), 'ud', [[1, 0], [0, 0]])
    B = Qu((2, 2), 'ud', [[0, 1], [0, 0]])
    C = Qu((2, 2), 'ud', [[0, 0], [1, 0]])
    D = Qu((2, 2), 'ud', [[0, 0], [0, 1]])
    names = "ABCD"
    ops = [A, B, C, D]
    for i in range(4):
        for j in range(4):
            op = ops[i] * ops[j]
            if op.norm() < EPSILON:
                continue
            k = ops.index(op)
            print(names[i], names[j], names[k])

#show()

def promote_algebra(algebra, G):
    A = Qu((2, 2), 'ud', [[1, 0], [0, 0]])
    B = Qu((2, 2), 'ud', [[0, 1], [0, 0]])
    C = Qu((2, 2), 'ud', [[0, 0], [1, 0]])
    D = Qu((2, 2), 'ud', [[0, 0], [0, 1]])
        
    basis = [A, B, C, D]
    v = decompose(basis, A)
    assert numpy.allclose(v, [1, 0, 0, 0])

    v = decompose(basis, A+B)
    assert numpy.allclose(v, [1, 1, 0, 0])

    #pauli = build_algebra("IXZY",
    #    "X*X=I Z*Z=I Y*Y=-I X*Z=Y Z*X=-Y X*Y=Z Y*X=-Z Z*Y=-X Y*Z=X")

    for op in G:
        v = decompose(basis, op)
        #g = Tensor({}, 1, algebra)
        g = algebra.get_zero(1)
        for i in range(algebra.dim):
            g = g + complex(v[i])*algebra.basis[i] # ARGH!
        yield g



class StabilizerCode(object):
    def __init__(self, algebra, ops):
        if type(ops) is str:
            ops = ops.split()
            ops = [algebra.parse(s) for s in ops]
        else:
            ops = list(ops)
        self.n = ops[0].grade
        for g in ops:
          for h in ops:
            assert g*h == h*g , "%s %s"%(g, h)
        self.ops = list(ops)
        self._P = None

    def get_projector(self):
        " Build projector onto codespace "
        if self._P is not None:
            return self._P
        G = mulclose(self.ops, verbose=False)
        #print("get_projector:", len(G))
        #P = (1./len(G))*reduce(add, G)
        P = reduce(add, G)
        self._P = P
        return P

    def __eq__(self, other):
        return self.get_projector() == other.get_projector()

    def __ne__(self, other):
        return self.get_projector() != other.get_projector()





def build_code(pauli):
    I = pauli.I
    X = pauli.X
    Y = pauli.Y
    Z = pauli.Z

    def mk_stab(s):
        s = s.replace(".", "I")
        sx = s.replace("1", "X")
        sz = s.replace("1", "Z")
        code = StabilizerCode(pauli, sx+sz)
        return code

    if argv.two:
        code = StabilizerCode(pauli, "XX ZZ")
        op = code.get_projector()
    elif argv.four:
        code = StabilizerCode(pauli, "XXII ZZII IIXX IIZZ")
        op = code.get_projector()
    elif argv.five:
        code = StabilizerCode(pauli, "XZZXI IXZZX XIXZZ ZXIXZ")
        op = (I@I@I@I@I+I@X@Z@Z@X-I@Z@Y@Y@Z-I@Y@X@X@Y
            +X@I@X@Z@Z-X@X@Y@I@Y+X@Z@Z@X@I-X@Y@I@Y@X
            -Z@I@Z@Y@Y+Z@X@I@X@Z+Z@Z@X@I@X-Z@Y@Y@Z@I
            -Y@I@Y@X@X-Y@X@X@Y@I-Y@Z@I@Z@Y-Y@Y@Z@I@Z)
        assert op == code.get_projector()
    elif argv.steane:
        code = StabilizerCode(pauli, "XXXXIII XXIIXXI XIXIXIX ZZZZIII ZZIIZZI ZIZIZIZ")
        op = code.get_projector()
    elif argv.rm:
        s = """
        1.1.1.1.1.1.1.1
        .11..11..11..11
        ...1111....1111
        .......11111111
        """
        code = mk_stab(s)
        op = code.get_projector()
    elif argv.toric:
        s = """
        11.11...
        .111..1.
        1...11.1
        """
        code = mk_stab(s)
        op = code.get_projector()
    else:
        op = argv.op
        if op:
            op = eval(op, locals())
    return op


def test_code():

    """
        Build some non-commuting polynomials in four variables: I X Z Y.
        Apply averaging operator to get invariant polynomials over 
        the "external" group which acts on (I X Z Y).
        Look for transversal gates built from the "internal" group, which
        is a finite subgroup of U(2).
    """

    pauli = build_algebra("IXZY",
        "X*X=I Z*Z=I Y*Y=-I X*Z=Y Z*X=-Y X*Y=Z Y*X=-Z Z*Y=-X Y*Z=X")

    I = pauli.I
    X = pauli.X
    Y = pauli.Y
    Z = pauli.Z

    op = build_code(pauli)
    #print(op)

    if op is not None:
        ops = [op]
        #print(op)
    
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
        ops = linear_independent(ops, algebra=pauli)
        print("dimension:", len(ops))

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
        # slow..... boo
    
        degree = op.grade
        P = op
    
        print("promote_pauli")
        _G = promote_pauli(pauli, G)
        GT = []
        count = 0
        for g in _G:
            print(".", end=" ", flush=True)
            op = reduce(matmul, [g]*degree)
            if op*P == P*op:
                count += 1
                print("+", end=" ", flush=True)
        print("commutes with", count, "transversal gates")

    def find_transversal_dense(op):
        I = Qu((2, 2), 'ud', [[1, 0], [0, 1]])
        X = Qu((2, 2), 'ud', [[0, 1], [1, 0]])
        Z = Qu((2, 2), 'ud', [[1, 0], [0, -1]])

        # which Y to use?
        #Y = Qu((2, 2), 'ud', [[0, 1.j], [-1.j, 0]]) 
        Y = X*Z
        
        P = op.evaluate({"I":I, "X":X, "Z":Z, "Y":Y})
        #print(P.shape)
        #P /= len(G)
    
        degree = op.grade
    
        if argv.show:
            print(op)

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
        find_transversal_dense(op)


def random_code(pauli, degree=4):

    "build a small random stabilizer code"

    I = pauli.I
    X = pauli.X
    Z = pauli.Z
    Y = pauli.Y

    ops = []
    negi = -reduce(matmul, [I]*degree)
    items = [I, X, Z, Y]
    itemss = [[I, X], [I, Z]]
    while len(ops) < degree-1:
        if argv.css:
            items = choice(itemss)
        op = [choice(items) for _ in range(degree)]
        op = reduce(matmul, op)
        for op1 in ops:
            if op1*op != op*op1:
                break
        else:
            #print(op)
            ops.append(op)
#            G = mulclose_fast(ops)
            G = mulclose(ops)
            if negi in G:
                ops = []

#    for op in ops:
#        print(op, end=" ")
#    print()
    code = StabilizerCode(pauli, ops)
    #G = mulclose_fast(ops)
    #for g in G:
    #    print(g, negi==g)

    P = code.get_projector()
#    print(P)

    return P


def test_gen_codes():
    pauli = build_algebra("IXZY",
        "X*X=I Z*Z=I Y*Y=-I X*Z=Y Z*X=-Y X*Y=Z Y*X=-Z Z*Y=-X Y*Z=X")

    I = pauli.I
    X = pauli.X
    Z = pauli.Z
    Y = pauli.Y

    n = argv.get("n", 4)
    m = argv.get("m", 2)
    items = [I, X, Z, 1.j*Y]

    gens = list(cross((items,)*n))
    gens = [reduce(matmul, op) for op in gens]
    g = gens[0]
    assert g*g==g # identity
    gens.pop(0)
    N = len(gens)
    print("gens:", len(gens))

    found = set()
    ops = []
    zero = pauli.get_zero(n)

    if m==0:
        pass

    if m==1:
      for i in range(N):
        ii = gens[i]
        G = mulclose([ii])
        if len(G)!=2**m:
            continue
        op = reduce(add, G)
        if op == zero:
            continue
        s = str(op)
        if s not in found:
            #print(s)
            found.add(s)
            ops.append(op)

    elif m==2:
      for i in range(N):
        ii = gens[i]
        for j in range(i+1, N):
          jj = gens[j]
          if ii*jj != jj*ii:
              continue
          rows = [ii, jj]
          G = mulclose(rows)
          if len(G)!=2**m:
              continue
          op = reduce(add, G)
          if op == zero:
              continue
          s = str(op)
          if s not in found:
              found.add(s)
              ops.append(op)

    elif m==3:

     for i in range(N):
      print("i =", i)
      ii = gens[i]
      for j in range(i+1, N):
        jj = gens[j]
        if ii*jj != jj*ii:
            continue
        for k in range(j+1, N):
            kk = gens[k]
            if ii*kk != kk*ii:
                continue
            if jj*kk != kk*jj:
                continue
            rows = [ii, jj, kk]
            G = mulclose(rows)
            if len(G)!=2**m:
                continue
            op = reduce(add, G)
            if op == zero:
                continue
            s = str(op)
            if s not in found:
                found.add(s)
                ops.append(op)

    print(len(ops))


def test_code2():

    """
        Build some non-commuting polynomials in four variables: I X Z Y,
        from projector onto codespace of stabilizer code.
        Apply averaging operator to get invariant polynomials over 
        the "external" group which acts on (I X Z Y).
        Look for transversal gates built from the "internal" group, which
        is a finite subgroup of U(2).
    """

    pauli = build_algebra("IXZY",
        "X*X=I Z*Z=I Y*Y=-I X*Z=Y Z*X=-Y X*Y=Z Y*X=-Z Z*Y=-X Y*Z=X")

    I = pauli.I
    X = pauli.X
    Z = pauli.Z
    Y = pauli.Y

    degree = argv.get("degree", 2)
    #op = build_code(pauli)
    op = random_code(pauli, degree)
    
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
            #print(p1)
        return p1

    G4 = argv.get("G4", "QT") # the "external" group
    if G4 is not None:
        G4 = eval(G4)
        print("|G4|=", len(G4))
        op = orbit_sum(G4, op)
    #print(op)
    if op==op.get_zero():
        return

    if op is None:
    
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
        ops = linear_independent(ops, algebra=pauli)
        print("dimension:", len(ops))
    else:
        ops = [op]

    # ------------------------------------------

    if not argv.find_transversal:
        return

    def show_spec(P):
        items = P.eigs()
        for val, vec in items:
            print(fstr(val), end=" ")
        print()

    G = eval(argv.get("G2", "Cliff")) # the "internal" group

    def find_transversal(op):
        # slow..... boo
    
        degree = op.grade
        P = op
    
        print("promote_pauli")
        _G = promote_pauli(pauli, G)
        GT = []
        count = 0
        for g in _G:
            print(".", end=" ", flush=True)
            op = reduce(matmul, [g]*degree)
            if op*P == P*op:
                count += 1
                print("+", end=" ", flush=True)
        print("commutes with", count, "transversal gates")

    def find_transversal_dense(op):
        I = Qu((2, 2), 'ud', [[1, 0], [0, 1]])
        X = Qu((2, 2), 'ud', [[0, 1], [1, 0]])
        Z = Qu((2, 2), 'ud', [[1, 0], [0, -1]])

        # which Y to use?
        #Y = Qu((2, 2), 'ud', [[0, 1.j], [-1.j, 0]]) 
        Y = X*Z
        
        P = op.evaluate({"I":I, "X":X, "Z":Z, "Y":Y})
        #print(P.shape)
        #P /= len(G)
    
        degree = op.grade
    
        if argv.show:
            print(op)

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
        find_transversal_dense(op)



def test_internal():

    r"""
        The internal group acts as P \mapsto g^{-1} P g.
        Average over this action to build invariant operators.
        Take as input a codespace projector.
        The average is not always a projector.
    """

    pauli = build_algebra("IXZY",
        "X*X=I Z*Z=I Y*Y=-I X*Z=Y Z*X=-Y X*Y=Z Y*X=-Z Z*Y=-X Y*Z=X")

    I = pauli.I
    X = pauli.X
    Z = pauli.Z
    Y = pauli.Y

    degree = argv.get("degree", 2)
    #op = build_code(pauli)
    P = random_code(pauli, degree)

    print(P)
    
    G = eval(argv.get("G2", "Tetra")) # the "internal" group
    n = len(G)
    G.build()

    # express each g as a sum of pauli's
    GP = [g for g in promote_pauli(pauli, G)]

    # transverse operators
    print("transverse operators")
    TG = [reduce(matmul, [g]*degree) for g in GP]

    print("averaging over group")
    Q = P.get_zero()
    for i in range(n):
        Q = Q + TG[G.inv[i]] * P * TG[i]
        write(".")
    print()

    print(Q)
    print(Q*Q)

    if argv.check:
        for i in range(n):
            Q1 = TG[G.inv[i]] * Q * TG[i]
            assert Q1 == Q


def test_molien():

    r"""
    """

    G = eval(argv.get("G2", "Tetra")) # the "internal" group
    n = len(G)
    #G.build()

    degree = argv.get("degree", 20)
    series = [0.]*(degree+1)

    for g in G:
        r = g.trace()
        #assert abs(r-round(r.real)) < EPSILON, r
        #r = int(round(r.real))
        #assert abs(r-r.real) < EPSILON, r
        #r = r.real

        for i in range(degree+1):
            # p = 1/(1-r*t)
            val = r**i
            series[i] += val
    #assert abs(sum(s%n for s in series))<EPSILON
    series = [s/n for s in series]
    #print(series)
    print(' '.join([fstr(s) for s in series]))




def test_internal_series():

    r"""
        The internal group acts as P \mapsto g^{-1} P g.
        Average over this action to build invariant operators.
        Take as input a codespace projector.
        The average is not always a projector.
    """

    pauli = build_algebra("IXZY",
        "X*X=I Z*Z=I Y*Y=-I X*Z=Y Z*X=-Y X*Y=Z Y*X=-Z Z*Y=-X Y*Z=X")

    I = pauli.I
    X = pauli.X
    Z = pauli.Z
    Y = pauli.Y

    G = eval(argv.get("G2", "Tetra")) # the "internal" group
    n = len(G)
    G.build()

    print("|G2| =", len(G))

    # express each g as a sum of pauli's
    PG = [g for g in promote_pauli(pauli, G)]

    # transverse operators
    print("transverse operators")
    degree = argv.get("degree", 2)
    TG = [reduce(matmul, [g]*degree) for g in PG]

    found = []
    for op in cross(([I, X, Z, Y],)*degree):
        P = reduce(matmul, op)

        Q = P.get_zero()
        for i in range(n):
            Q = Q + TG[G.inv[i]] * P * TG[i]

        if Q==Q.get_zero():
            continue
        print(P)
        print(Q)
        found.append(Q)

    print("found ops:", len(found))
    if not found:
        return
    found = linear_independent(found, pauli.construct)
    print("dimension:", len(found))



def allperms(items):
    items = tuple(items)
    if len(items)<=1:
        yield items
        return
    n = len(items)
    for i in range(n):
        for rest in allperms(items[:i] + items[i+1:]):
            yield (items[i],) + rest


def to_dense(op):
    I = Qu((2, 2), 'ud', [[1, 0], [0, 1]])
    X = Qu((2, 2), 'ud', [[0, 1], [1, 0]])
    Z = Qu((2, 2), 'ud', [[1, 0], [0, -1]])

    # which Y to use?
    #Y = Qu((2, 2), 'ud', [[0, 1.j], [-1.j, 0]]) 
    Y = X*Z

    op = op.subs({"I":I, "X":X, "Z":Z, "Y":Y})
    return op

        
def test_internal_series_fast():

    r"""
        The internal group acts as P \mapsto g^{-1} P g.
        Average over this action to build invariant operators.
        Take as input a codespace projector.
        The average is not always a projector.
        Make it faster by using the action of the symmetric group
        on the tensor factors.
    """

    def show_spec(P):
        items = P.eigs()
        for val, vec in items:
            print(fstr(val), end=" ")
        print()

    use_pauli = argv.get("use_pauli", True)

    if use_pauli:
        names = "IXZY"
        algebra = build_algebra(names,
            "X*X=I Z*Z=I Y*Y=-I X*Z=Y Z*X=-Y X*Y=Z Y*X=-Z Z*Y=-X Y*Z=X")
        promote = promote_pauli

    else:
        # the Algebra of 2x2 E_{ij} indicator matrices
        dim = 4
        names = "ABCD"
        A, B, C, D = range(dim)
        coefs = {}
        coefs[A, A, A] = 1.
        coefs[A, B, B] = 1.
        coefs[B, C, A] = 1.
        coefs[B, D, B] = 1.
        coefs[C, A, C] = 1.
        coefs[C, B, D] = 1.
        coefs[D, C, C] = 1.
        coefs[D, D, D] = 1.
        algebra = Algebra(dim, names, coefs)
        promote = promote_algebra

    basis = [getattr(algebra, name) for name in names]

    G = eval(argv.get("G2", "Tetra")) # the "internal" group
    n = len(G)
    G.build()

    print("|G2| =", len(G))

    # express each g as a sum of pauli's
    PG = [g for g in promote(algebra, G)]

    if 0:
        for g in [X, Z, Y]:
            print("%s: "%g, end=" ")
            for i in range(n):
                R = PG[G.inv[i]] * g * PG[i]
                print(R, end=" ")
            print()
        return

    def is_abelian(op):
        abelian = True
        terms = op.get_terms()
        for i, a in enumerate(terms):
            for b in terms[i+1:]:
                if a*b != b*a:
                    abelian = False
                    break
        return abelian

    # transverse operators
    print("transverse operators")
    degree = argv.get("degree", 2)
    TG = [reduce(matmul, [g]*degree) for g in PG]

    found = []
    opis = set(cross((list(range(len(basis))),)*degree))
    weight = argv.weight

    if argv.full:
        weight = degree

    if weight is not None:
        opis = [opi for opi in opis if degree-opi.count(0)==weight]

    print("opis:", len(opis))

    perms = list(allperms(list(range(degree))))

    if degree>4:
        debug = write
    else:
        def debug(*args):
            pass

    while opis:
        debug("[%d]"%len(opis))
        opi = iter(opis).__next__()

        op = [basis[i] for i in opi]
        P = reduce(matmul, op)
        debug("[%s]"%P)

        # any one of these guys will generate the same Q operator
        singletons = set([opi])

        # find Q as a sum over the action of the group on P
        Q = P.get_zero()
        for i in range(n):
            R = TG[G.inv[i]] * P * TG[i]
            nnz = R.nnz()
            if nnz == 0:
                continue
            assert nnz>=1
            if nnz==1:
                keys = R.get_keys()
                key = keys[0]
                singletons.add(key)
                #if key!=opi and key in opis:
                #    opis.remove(key)
                #    debug("x")
#                debug("|")
#                print([str(k) for k in R.get_keys()], end="")
            Q = Q + R
            #print("\t", R)
            debug(".")
        debug("\n")

        if Q==Q.get_zero():
            #debug("==0 ")
            for perm in perms:
                for opi in singletons:
                    opj = tuple(opi[perm[i]] for i in range(degree))
                    if opj in opis:
                        opis.remove(opj)
                        #debug("x")
            #debug("\n")
            continue

        for perm in perms:
            opj = tuple(opi[perm[i]] for i in range(degree))
            if opj not in opis:
                continue
            _P = P.permute(perm)
            _Q = Q.permute(perm)
            print("%s: %s" %(_P, _Q))
            found.append(_Q)
            opis.remove(opj)

        if argv.show_spec:
            A = to_dense(Q)
            show_spec(A)

        if is_abelian(Q):
            print(" ========= ABELIAN ============ ")
        else:
            print()
        
    print("found ops:", len(found))
    if not found:
        return
    basis = linear_independent(found, algebra.construct)
    print("dimension:", len(basis))
    if argv.show_basis:
        for op in basis:
            print(op, "abelian" if is_abelian(op) else "")

    write("building algebra of invariants...")

    all_keys = []
    for A in basis:
        all_keys += A.get_keys()
    all_keys = list(set(all_keys))
    all_keys.sort()

    def fixup(val):
        if abs(val.imag)<EPSILON:
            val = val.real
            if abs(val-round(val))<EPSILON:
                val = int(round(val))
        return val
    Et = []
    N = len(basis)
    for e in basis:
        Et.append([fixup(e[key]) for key in all_keys])
    Et = numpy.array(Et, dtype=float)
    E = Et.transpose()
    Einv = numpy.linalg.pinv(E, EPSILON)
    #print(E)
    #print(Einv)
    #print(numpy.dot(Einv, E))

    struct = numpy.zeros((N, N, N))

    for i, A in enumerate(basis):
      for j, B in enumerate(basis):
        C = A*B
        #print("(%s) * (%s) = %s" %(A, B, C))
        v = [fixup(C[key]) for key in all_keys]
        cij = numpy.dot(Einv, v)
        #print(cij)
        struct[i, j] = cij
        if argv.test_closed:
            for k in range(n):
                D = TG[G.inv[k]] * C * TG[k]
                assert D == C 

    A = AssocAlg(struct)
    #I = A.unit
    #print(A.construct(I, basis))

    write("\n")

    ZA_basis = A.find_center()
    print("center:", len(ZA_basis))

    for x in ZA_basis:
        print(x)
        x = A.construct(x, basis)
        #print(x)
        for y in basis:
            assert x*y == y*x

    code = None
    if degree==5:
        code = StabilizerCode(algebra, "XZZXI IXZZX XIXZZ ZXIXZ")
        code = code.get_projector()
    elif degree==4:
        code = StabilizerCode(algebra, "ZZII IZZI IIZZ XXXX")
        code = code.get_projector()
    elif degree==3:
        code = StabilizerCode(algebra, "ZZI IZZ XXX")
        code = code.get_projector()
    
    ZA = A.subalgebra(ZA_basis)
    #print("unit:", ZA.unit)
    print("projectors:")
    central_idem = []
    for x in ZA.find_idempotents(1000):

        x = [fixup(xx) for xx in x]

        x1 = ZA.construct(x, ZA_basis) # x1 is in A
        P = A.construct(x1, basis) # P is in pauli
        if P in central_idem:
            continue

        central_idem.append(P)
        print(P)
        err = (P*P - P).norm()
        if abs(err) > EPSILON:
            print("error:", err)

        #if code is not None:
        #    print(P*code == code)

        if str(P) == "I"*degree:
            assert 0
            continue

        block = []
        for e in A:
            f = A.mul(x1, e)
            #print(e, "-->", f)
            block.append(f)
        block = numpy.array(block)
        #print(block)
        block = row_reduce(block, truncate=True)
        #print(block)
        print("rank:", len(block))

        #sub = A.subalgebra(block)
        #if sub.N == 2:
        #    for x in sub.find_idempotents(100, verbose=True):
        #        print("FOUND:", x)

        #mod = A.mod(block)
        #for x in mod.find_idempotents(100):

        #if len(central_idem) == len(ZA_basis):
        #    break


    return

    trials = argv.get("trials", 0)
    Ps = []
    for x in A.find_idempotents(trials):
        P = A.construct(x, basis)
        err = (P*P - P).norm()
        if abs(err) > EPSILON:
            print("error:", err)
        Ps.append(P)
        #P = to_dense(P)
        #show_spec(P)



def parse_op(algebra, s):
    terms = []
    idx = 1
    s0 = s
    while idx < len(s):
        if s[idx]=="+" or s[idx]=="-":
            term = s[:idx]
            terms.append(term)
            s = s[idx:]
            idx = 0
        idx += 1
    terms.append(s)
    assert ''.join(terms) == s0
    ops = []
    for term in terms:
        if "*" in term:
            term = term.split("*")[1]
        op = algebra.parse(term)
        ops.append(op)
    return ops


def find_commutative_invariants():

    pauli = build_algebra("IXZY",
        "X*X=I Z*Z=I Y*Y=-I X*Z=Y Z*X=-Y X*Y=Z Y*X=-Z Z*Y=-X Y*Z=X")

    name = argv.next()
    print("loading", name)
    f = open(name)
    for line in f:
        #print(line)
        line = line.strip()
        flds = line.split(" ")
        assert len(flds)==2
        op = flds[1]
        ops = parse_op(pauli, op)
        N = len(ops)
        abelian = True
        #for op in ops:
        #    print(op, repr(op))
        for i in range(N):
          for j in range(i+1, N):
            a, b = ops[i], ops[j]
            if a*b != b*a:
                abelian = False
        if abelian:
            print(op)


def main():
    pass


if __name__ == "__main__":

    build()

    _seed = argv.seed
    if _seed is not None:
        numpy.random.seed(_seed)
        seed(_seed)

    name = argv.next() or "main"

    if argv.profile:
        import cProfile as profile
        profile.run("%s()"%name)
    else:
        fn = eval(name)
        fn()


