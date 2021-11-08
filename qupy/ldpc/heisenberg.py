#!/usr/bin/env python3

"""

An attempt to implement the ideas in 
The Weil representation in characteristic two
Shamgar Gurevich, Ronny Hadani
https://www.maths.ed.ac.uk/~v1ranick/papers/gurevichhadani2.pdf

See also:
https://kups.ub.uni-koeln.de/50465/1/dissertation_heinrich.pdf

"""

print
from random import shuffle

import numpy
from numpy import dot, alltrue, zeros, array, identity, kron, concatenate


from qupy.ldpc import asymplectic
from qupy.util import mulclose_fast, mulclose_hom, mulclose_names
from qupy.smap import SMap
from qupy.argv import argv


int_scalar = numpy.int8


_cache = {}
def symplectic_form(n):
    if n in _cache:
        return _cache[n]
    F = numpy.zeros((2*n, 2*n), dtype=int_scalar)
    I = numpy.identity(n, dtype=int_scalar)
    F[:n, n:] = I
    F[n:, :n] = I
    _cache[n] = F
    return F


class Heisenberg(object):

    def __init__(self, v, phase=0):
        assert 0<=phase<4 # Z/4
        n = len(v)
        assert n%2 == 0
        n //= 2
        self.n = n
        self.v = array(v, dtype=int_scalar) % 2
        self.phase = phase

    def __str__(self):
        return "%s[%s]"%(self.phase, self.v)
    __repr__ = __str__

    def __eq__(u, v):
        assert u.n == v.n
        return u.phase == v.phase and alltrue(u.v == v.v)

    def __hash__(u):
        key = (u.v.tobytes(), u.phase)
        return hash(key)

    def sy(u, v):
        assert u.n == v.n
        n = u.n
        F = symplectic_form(n)
        a = dot(u.v, dot(F, v.v)) % 2
        return a

    @property
    def X(u):
        return u.v[:u.n]

    @property
    def Z(u):
        return u.v[u.n:]

    def beta(u, v):
        assert u.n == v.n
        n = u.n
        #u, v = u.v, v.v
        #F = symplectic_form(n)
        #u_Z = u[n:] 
        #v_X = v[:n] 
        d = dot(u.Z, v.X) % 2
        return 2*d

    def __neg__(u):
        return Heisenberg(u.v.copy(), (u.phase+2)%4)
        
    def __mul__(u, v):
        assert u.n == v.n
        phase = (u.beta(v) + u.phase + v.phase) % 4
        w = (u.v + v.v) % 2
        return Heisenberg(w, phase)

    def __matmul__(u, v):
        phase = (u.phase + v.phase) % 4
        v = concatenate((u.X, v.X, u.Z, v.Z))
        return Heisenberg(v, phase)

    def __pow__(self, count):
        assert count >= 0
        if count == 0:
            return Heisenberg([0]*self.n*2, 0)
        if count == 1:
            return self
        A = self
        while count > 1:
            A = self * A
            count -= 1
        return A





def main():

    I = Heisenberg([0, 0])
    w = Heisenberg([0, 0], 1)
    X = Heisenberg([1, 0])
    Z = Heisenberg([0, 1])
    Y = Heisenberg([1, 1], 1)
    XZ = Heisenberg([1, 1])
    ZX = Heisenberg([1, 1], 2)

    assert X.sy(Z) == 1
    assert Z.sy(X) == 1

    assert X.beta(Z) == 0
    assert Z.beta(X) == 2

    assert XZ != ZX
    assert XZ == -ZX
    assert X*Z == XZ
    assert Z*X == ZX
    assert Y == w*X*Z

    assert w != I
    assert w**2 != I
    assert w**3 != I
    assert w**4 == I

    XI = X@I
    ZI = Z@I
    IX = I@X
    IZ = I@Z
    XZ = X@Z
    ZX = Z@X

    assert XI*IX == X@X
    assert ZI*XI == -XI*ZI
    assert XZ * ZX == ZX * XZ

    G = mulclose_fast([X, Y, Z])
    assert w in G
    assert len(G) == 16

    # test beta is a 2-cocycle
    beta = lambda u, v : u.beta(v)
    for g in G:
      for h in G:
        for k in G:
            lhs = (beta(g, h) - beta(h, k))%4
            rhs = (beta(g, h*k) - beta(g*h, k))%4
            print(lhs, rhs, end=" ")
            assert lhs == rhs


if __name__ == "__main__":

    main()



