#!/usr/bin/env python3

import sys
import math
import numpy
from random import choice, randint, seed, shuffle
from functools import reduce
seed(0)

from operator import mul

from qupy.abstract import Space
from qupy.dense import Qu, Gate, Vector
from qupy.dense import genidx, bits, is_close, on, off, scalar
from qupy.dense import commutator, anticommutator

from qupy.test import test

epsilon = 1e-10
r2 = math.sqrt(2)

def test_vector():

    v = Vector([1,2])
    w = Vector([4,1])
    assert (v+w).is_close( Vector([5, 3]) )
    assert (2*v).is_close( Vector([2, 4]) )
    assert (v*2).is_close( Vector([2, 4]) )
    assert (v/2).is_close( Vector([0.5, 1]) )
    w *= 2
    assert w.is_close( Vector([8, 2]) )
    _w = w
    _w /= 2
    assert _w.is_close( Vector([4, 1]) )
    assert _w is w

    v = Vector([1,1])
    assert is_close( v.norm(), r2 )
    assert is_close( v.normalized().norm(), 1. )
    assert is_close( v.norm(), r2 )
    v.normalize()
    assert is_close( v.norm(), 1. )


def test_setitem():

    A = Qu((2,)*3, 'uuu')
    A[:] = bits('010')
    #A[0, :] # XXX TODO


def test_bits():

    assert bits('000').decode() == '000'
    assert bits('001').decode() == '001'
    assert bits('010').decode() == '010'
    assert bits('100').decode() == '100'
    assert bits('110').decode() == '110'


def test_tensor_product():

    A = Gate((2, 2))
    A[0, 0] = 1.
    A[0, 1] = 2.
    A[1, 0] = 3.
    A[1, 1] = 4.

    B = Gate((2, 2))
    B[0, 0] = 5.
    B[0, 1] = 6.
    B[1, 0] = 7.
    B[1, 1] = 8.

    AB = A*B
    for i, j, k, l in genidx((2, 2, 2, 2)):
        assert AB[i, j, k, l] == A[i, j] * B[k, l]

    AB = AB.contract(1, 2)
    for i, j in genidx((2, 2)):
        #print i, j, AB[i, j], "=>", sum(A[i, k] * B[k, j] for k in (0, 1))
        assert AB[i, j] == sum(A[i, k] * B[k, j] for k in (0, 1))

    assert on.shape == (2,)

    assert (on*on).is_close(bits('11'))

    c = on*off
    assert c.shape == (2, 2)

    c = on*off*on*on
    assert c.shape == (2,)*4

    assert c.is_close(bits('1011'))

    for i0 in (0, 1):
        for i1 in (0, 1):
            for i2 in (0, 1):
                for i3 in (0, 1):
                    r = c[i0, i1, i2, i3]
                    assert (r != 0.) == ((i0, i1, i2, i3) == (True, False, True, True))

    x = Gate.H.apply(off)
    assert x.space == Space((2,), 'u'), str(x)

    y = (Gate.H * off).contract(1, 2)
    assert x.is_close(y)

    A = Gate.H * Gate.I
    #print A.shape, A.valence

    x = bits('00')
    #print x.shape, x.valence


def test_space_pipe():

    a = Space((2, 2, 2, 2), 'udud')
    b = Space((2, 2, 2, 3), 'uddu')
    perm = a.unify(b)
    assert perm is None

    a = Space((2, 2, 2, 2), 'udud')
    b = Space((2, 2, 2, 2), 'uddu')
    perm = a.unify(b)
    assert perm == [0, 1, 3, 2]

    a = Space(2, 'u')
    b = Space(2, 'd')
    assert b|a == Space((), '')

    a = Space((2, 3), 'ud')
    b = Space((3, 4), 'ud')
    assert a|b == Space((2, 4), 'ud')

    a = Space((2, 2, 2, 2), 'udud')
    assert a|a == a, a|a

    for i in range(100):
        n = randint(0, 5)
        valence = list('u'*n + 'd'*n)
        shuffle(valence)
        valence = ''.join(valence)
        a = Space((2,)*(2*n), valence)
        assert a|a == a, (a|a, a)


def test_pipe_op():

    a = Qu(2, 'u', [0, 1])
    b = Qu(2, 'd', [1, 2])
    c = b | a # dot product!
    assert type(c) is scalar
    assert is_close(c, 2)

    A = Qu((2, 2), 'ud')
    v = Qu(2, 'u')
    u = A | v
    assert u.space == v.space

    # The pipe operator: valence

    A = Gate.I * Gate.X
    B0 = A | A
    B1 = A*A

    assert B1.valence == "udududud"
    valence = list(B1.valence)
    B1 = B1.contract(1, 4)
    assert B1.valence == "uuddud"
    B1 = B1.contract(2, 4)
    assert B1.valence == "uudd"

    B1.permute([0, 2, 1, 3])

    assert B0.is_close(B1)

    assert (1.j|A).is_close(1.j*A)

    B = A.clone()
    A |= A

    assert A.is_close(B|B)


def test_pipe_tensor():

    I, X, Y, Z, H = Gate.I, Gate.X, Gate.Y, Gate.Z, Gate.H
    CN = Gate.CN

    assert I*I|I*I == I*I

    A = (X*I*H | I*Z*I)
    B = (X*Z*H)

    C = A.unify(B)

    #perm = A.space.get_perm(B.space)
    #print A.valence, B.valence
    #print A.v.transpose(perm).shape
    #print [B.valence[i] for i in perm]

    assert X*I*H | I*Z*I == X*Z*H
    CN*I | I*CN


def test_gate_truth():

    # FLIP truth table (NOT gate)
    A = Gate.X
    assert (A | bits('0')).is_close(bits('1'))
    assert (A | bits('1')).is_close(bits('0'))

    # Identity * FLIP truth table
    A = Gate.I * Gate.X
    assert (A | bits('00')).is_close(bits('01'))
    assert (A | bits('01')).is_close(bits('00'))
    assert (A | bits('10')).is_close(bits('11'))
    assert (A | bits('11')).is_close(bits('10'))

    # Controlled-NOT truth table
    A = Gate.CN
    assert (A | bits('00')).is_close(bits('00'))
    assert (A | bits('01')).is_close(bits('01'))
    assert (A | bits('10')).is_close(bits('11'))
    assert (A | bits('11')).is_close(bits('10'))

    A = Gate.SWAP
    assert (A | bits('00')).is_close(bits('00'))
    assert (A | bits('01')).is_close(bits('10'))
    assert (A | bits('10')).is_close(bits('01'))
    assert (A | bits('11')).is_close(bits('11'))


def test_gate_identities():

    X, Y, Z, H, I = Gate.X, Gate.Y, Gate.Z, Gate.H, Gate.I

    zero = Qu((2, 2), 'ud')

    assert (X | X) == I
    assert (Y | Y) == I
    assert (Z | Z) == I

    assert anticommutator(X, Z) == zero
    assert anticommutator(Y, Z) == zero
    assert anticommutator(X, Y) == zero

    assert (Z|X) == 1.j*Y
    assert (X|Y) == 1.j*Z
    assert (Y|Z) == 1.j*X

    CZ = Z.control()
    A = (I*H)|CZ|(I*H)
    CX = X.control()
    assert A == CX

    assert CZ|CX == (Z|X).control()
    assert CX|CZ == (X|Z).control()

    assert (X.control() | (H*I)) | ((H*I) | X.control()) == I*I

    assert H==~H
    assert H|~H==I
    assert H|X|~H==Z


def test_pow():

    for A in (Gate.I, Gate.Z):
        assert A**1 == A
        assert A**2 == A*A
        assert A**3 == A*A*A

def test_swap():

    assert bits('0100').swap((1, 2)).decode() == '0010'
    assert bits('0101').swap((1, 2), (0, 3)).decode() == '1010'


def test_gate_properties():

    A = Gate((2, 2))
    A[0, 1] = 1.
    assert not A.is_unitary()
    assert not A.is_hermitian()

    for A in Gate.I, Gate.X, Gate.Y, Gate.Z, Gate.H:
        assert A.is_unitary()

    for A in Gate.I, Gate.X, Gate.Z, Gate.H:
        assert A.is_hermitian()


def test_bell_basis():

    raise test.Skip

    A = (Gate.H * Gate.I) | Gate.CN

    print()
    xs = Qu.bell_basis(2)
    for x in xs:
        print(x.shortstr())

    print()
    for i, word in enumerate(['00', '01', '10', '11']):
        x = bits(word)
        print((A|x).shortstr())


def test_transpose():

    a = Qu.random(2, 'u')
    r = ~a | a
    assert is_close(r, a.norm()**2)

    v = a.v[:]
    v.shape = v.shape + (1,)
    A = Qu(v.shape, 'ud', v)

    r = ~A | A
    assert is_close(r[0, 0], a.norm()**2)


def test_control():

    X, Y, Z, I, CN = Gate.X, Gate.Y, Gate.Z, Gate.I, Gate.CN

    assert X.control() == CN

    A = Z.control()
    assert A | (off * I) == off * I
    assert A | (on * I) == on * Z


def test_pure():
    v = Qu.random((2,), 'u')
    v /= v.norm()
    A = v * ~v
    A = Gate.promote(A)

    assert is_close(A.trace(), 1.)
    assert A.is_pure()

    A = 0.5 * Gate.dyads[0] + 0.5 * Gate.dyads[1]
    A = Gate.promote(A)
    assert is_close(A.trace(), 1.)
    assert not A.is_pure()


def test_adjoint():

    for A in Gate.I, Gate.X, Gate.Y, Gate.Z, Gate.H:
        B = A | ~A
        assert B.is_close(Gate.I)
        B = ~A | A
        assert B.is_close(Gate.I)

    A = Qu.random((2, 3), 'ud')

    assert (~A).shape == (2, 3)
    assert (~A).valence == 'du'

    H = ~A | A
    assert Gate.promote(H).is_hermitian()

    B = Qu.random((4, 2), 'ud')
    E = Qu.random((3, 2, 2), 'udu')
    u = Qu.random(2, 'u')
    v = Qu.random(3, 'd')

    for i in range(100):
        word = []
        for j in range(randint(1, 4)):
            word.append(choice((A, B, E, u, v)))
    
        lhs = ~reduce(mul, word)
        rhs = reduce(mul, [~x for x in word])
    
        assert lhs.space == rhs.space, (lhs.space, rhs.space)

        assert lhs.is_close(rhs)


def test_random():

    A = Qu.random((2, 2), 'ud')

    A = Gate.random_hermitian(4)
    assert A.is_hermitian()

    A = Gate.random_unitary(4)
    assert A.is_unitary()


def test_evolve():

    t = 1.

    H = Gate.random_hermitian(2)
    U = H.evolution_operator(t)
    W = H.evolution_operator(t/2)

    #print (W|W).shortstr()

    assert (W|W).is_close(U)

    v = Qu.random(2, 'u')
    v.normalize()

    #print
    w = U|v
    #print w.shortstr()

    x = W|v
    x = W|x
    #print x.shortstr()
    assert w.is_close(x)


def test_bitflip_code():

    x0 = bits('000')
    x1 = bits('111')

    A = Qu((2,)*4, "uduu")

    A[:, 0, :, :] = x0
    A[:, 1, :, :] = x1

    assert ( A | off ).is_close(x0)
    assert ( A | on ).is_close(x1)

    B = Gate.CN
    B = B | (Gate.I * off)
    B = B * off

    C = Gate.CN * Gate.I
    assert (C | bits('000')).decode() == '000'
    assert (C | bits('100')).decode() == '110'
    assert (C | bits('111')).decode() == '101'
    C = C.swap((2, 4), (3, 5)) # yuck...
    assert (C | bits('000')).decode() == '000'
    assert (C | bits('100')).decode() == '101'
    assert (C | bits('111')).decode() == '110'

    B = C | B

    assert B.is_close( A )

    x = Qu.random(2, 'u')
    x.normalize()

    y = A | x # encode

    # now entangle with environment...

    env = Qu.random(2, 'u')

    z = y * env

    space = z.space * ~z.space
    U = Qu.random_unitary(space)

    z1 = U | z


def test_shor_code():

    x0 = (1./(2*r2)) * (bits('000') + bits('111'))\
        * (bits('000') + bits('111')) \
        * (bits('000') + bits('111'))
    x1 = (1./(2*r2)) * (bits('000') - bits('111')) \
        * (bits('000') - bits('111')) \
        * (bits('000') - bits('111'))

    A = Qu((2,)*10, 'u'*9 + 'd')

    A[(slice(None),)*9+(0,)] = x0
    A[(slice(None),)*9+(1,)] = x1

    x = Qu.random(2, 'u')
    x.normalize()

    y = A | x # encode

    # now entangle with environment...

    env = Qu.random(2, 'u')

    z = y * env

    return 

    # this takes a while...

    rank = z.rank
    space = Space(2**rank, 'u')
    H = Qu.random_hermitian(space*~space)
    print("H:", H.space)
    U = H.evolution_operator(1.)
    print("U:", U.space, "rank:", U.rank)

    op = z.get_flatop()
    z = op.do(z)

    z1 = U | z

    if 0:
        space = z.space * ~z.space
    #    U = Qu.random_unitary(space)
        H = Qu.random_hermitian(space)
        print("H:", H.space)
        U = H.evolution_operator(1.)
        print("U:", U.space, "rank:", U.rank)

        z1 = U | z



def test_flatten():

    shape = (2, 3, 4)
    valence = 'uuu'
    A = Qu.random(shape, valence)
    op = A.get_flatop()
    B = op.do(A)
    assert B.shape == (2*3*4,), B.shape
    assert B.valence == 'u'


    shape = (2, 3)
    valence = 'ud'
    a = Qu.random(shape, valence)
    op = a.get_flatop()
    assert op.do(a).is_close(a)

    shape = (2, 3, 4, 5)
    valence = "udud"
    a = Qu.random(shape, valence)

    op = a.get_flatop()

    b = op.do(a)

    c = op.undo(b)

    assert c.is_close(a)


def test_flatten_exp():

    n1, n2 = 3, 4

    H_1 = Gate.random_hermitian(n1)
    H_2 = Gate.random_hermitian(n2)

    H = H_1 * Gate.identity(n2) + Gate.identity(n1) * H_2

    op = H.get_flatop()
    H_flat = op.do(H)
    H_flat = Gate.promote(H_flat)

    t = 1.
    U_flat = H_flat.evolution_operator(t)

    U_1 = H_1.evolution_operator(t)
    U_2 = H_2.evolution_operator(t)

    U = U_1 * U_2

    U_unflat = op.undo(U_flat)

    assert U.is_close(U_unflat)

    U_unflat = H.evolution_operator(t)

    assert U.is_close(U_unflat)







