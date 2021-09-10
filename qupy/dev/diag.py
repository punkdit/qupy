#!/usr/bin/env python3

from functools import reduce
from operator import matmul as tensor

import numpy
scalar = numpy.complex128

from qupy.util import mulclose_fast, mulclose_find


EPSILON = 1e-8

class Diag(object):
    """
    A _diagonal matrix
    """
    def __init__(self, diag):
        N = len(diag)
        diag = numpy.array(diag, dtype=scalar)
        assert diag.shape == (N,)
        self.diag = diag
        self.N = N

    def __str__(self):
        return "Diag(%s)"%(self.diag,)
    __repr__ = __str__

    def __len__(self):
        return self.N

    def __mul__(self, other):
        assert isinstance(other, Diag)
        diag = self.diag * other.diag
        return Diag(diag)

    def __matmul__(self, other):
        diag = numpy.outer(self.diag, other.diag)
        diag.shape = (self.N * other.N,)
        return Diag(diag)

    def __eq__(self, other):
        assert self.N == other.N
        return numpy.allclose(self.diag, other.diag)

    def get_cliff(self, level=2):
        assert level==2
        #phases = [1j**i for i in range(2**level)]
        diag = []
        for val in self.diag:
            for i in range(2**level):
                #print(1j**i, end=" ")
                if abs(val - 1j**i) < EPSILON:
                    diag.append(i)
                    break
            else:
                assert 0, "value %s not found" % (val,)
            #print()
        return Cliff(diag, level)
            
    def inner(self, other):
        assert self.N == other.N
        r = numpy.dot(self.diag, other.diag)
        return r


class Cliff(object):
    """
    _diagonal clifford (or more general than that?) at a level.
    """
    def __init__(self, diag, level=2):
        assert 0<level
        N = len(diag)
        diag = numpy.array(diag, dtype=int)
        diag = diag % (2**level)
        assert diag.shape == (N,)
        self.diag = diag
        self.N = N
        self.level = level

    def __str__(self):
        return "Cliff(%s)"%(self.diag,)

    __repr__ = __str__

    def __len__(self):
        return self.N

    def __mul__(self, other):
        assert isinstance(other, Cliff)
        assert self.N == other.N
        assert self.level == other.level
        diag = self.diag + other.diag
        return Cliff(diag)

    def __matmul__(self, other):
        assert isinstance(other, Cliff)
        assert self.level == other.level
        diag = numpy.add.outer(self.diag, other.diag)
        diag.shape = (self.N * other.N,)
        return Cliff(diag)

    def __eq__(self, other):
        assert isinstance(other, Cliff)
        assert self.N == other.N
        assert self.level == other.level
        return numpy.allclose(self.diag, other.diag)

    def __hash__(self):
        return hash(tuple(self.diag)) # .. probably slow, use tostring ...

    def inner(self, other):
        assert isinstance(other, Cliff)
        assert self.N == other.N
        assert self.level == other.level
        r = numpy.dot(self.diag, other.diag)
        #r %= 2**self.level
        return r


def test_diag():

    I = Diag([1, 1])
    Z = Diag([1, -1])
    S = Diag([1, 1j])
    Si = Diag([1, -1j])
    CZ = Diag([1, 1, 1, -1])
    II = Diag([1, 1, 1, 1])

    assert I == I
    assert I != Z
    assert Z*Z == I
    assert S*S == Z
    assert S*Si == I

    assert I@I == II
    assert CZ * CZ == II

    SI = S@I

    ops = [II, Z@I, I@Z, S@I, I@S, Z@S, S@Z, CZ]
    for a in ops:
      for b in ops:
        r = a.inner(b)
        #print("%6s" % r, end=" ")
      #print()
    # ???



def test_cliff():

    I = Cliff([0, 0])
    Z = Cliff([0, 2])
    S = Cliff([0, 1])
    Si = Cliff([0, 3])
    CZ = Cliff([0, 0, 0, 2])
    II = Cliff([0, 0, 0, 0])

    assert I == I
    assert I != Z
    assert Z*Z == I
    assert S*S == Z
    assert S*Si == I

    
    assert I@I == II
    assert CZ * CZ == II

    assert S@I == Cliff([0, 0, 1, 1])



def test_target():
    i = 1j
    target = Diag([
         1,  1,  i, -i,  i, -i,  1,  1,
         1,  1,  i, -i,  i, -i,  1,  1,
         1,  1, -i,  i,  i, -i, -1, -1,
         1,  1, -i,  i,  i, -i, -1, -1,
         i,  i, -1,  1, -1,  1,  i,  i,
         i,  i, -1,  1, -1,  1,  i,  i,
        -i, -i, -1,  1,  1, -1,  i,  i,
        -i, -i, -1,  1,  1, -1,  i,  i,
         1, -1,  i,  i,  i,  i,  1, -1,
        -1,  1, -i, -i, -i, -i, -1,  1,
         1, -1, -i, -i,  i,  i, -1,  1,
        -1,  1,  i,  i, -i, -i,  1, -1,
         i, -i, -1, -1, -1, -1,  i, -i,
        -i,  i,  1,  1,  1,  1, -i,  i,
        -i,  i, -1, -1,  1,  1,  i, -i,
         i, -i,  1,  1, -1, -1, -i,  i,
        -i, -i,  1, -1, -1,  1,  i,  i,
         i,  i, -1,  1,  1, -1, -i, -i,
         i,  i,  1, -1,  1, -1,  i,  i,
        -i, -i, -1,  1, -1,  1, -i, -i,
         1,  1,  i, -i, -i,  i, -1, -1,
        -1, -1, -i,  i,  i, -i,  1,  1,
         1,  1, -i,  i, -i,  i,  1,  1,
        -1, -1,  i, -i,  i, -i, -1, -1,
        -i,  i,  1,  1, -1, -1,  i, -i,
        -i,  i,  1,  1, -1, -1,  i, -i,
         i, -i,  1,  1,  1,  1,  i, -i,
         i, -i,  1,  1,  1,  1,  i, -i,
         1, -1,  i,  i, -i, -i, -1,  1,
         1, -1,  i,  i, -i, -i, -1,  1,
         1, -1, -i, -i, -i, -i,  1, -1,
         1, -1, -i, -i, -i, -i,  1, -1,
    ])

    k = 8
    N = 2**k
    assert len(target) == N
    
    #target = Diag(target.diag[:8])
    target = target.get_cliff()

    I = Cliff([0, 0])
    phase = Cliff([1])
    Z = Cliff([0, 2])
    S = Cliff([0, 1])
    Si = Cliff([0, 3])
    CZ = Cliff([0, 0, 0, 2])
    II = Cliff([0, 0, 0, 0])

    #print( I.inner(I) )
    #assert I.inner(I) == 2

    for i in range(k):
        ops = [I]*k
        ops[i] = S
        op = reduce(tensor, ops)
        #print(op.inner(target)) 

    op = reduce(tensor, [S, I, S, I, I, S, S, I])
    #print(op.inner(target)) 

    #print(target)
    #print(op)

    p = Cliff([1]*N) # phase
    gens = [p]
    for i in range(k):
        ops = [I]*k
        ops[i] = S
        op = reduce(tensor, ops)
        gens.append(op)

    for i in range(k):
      for j in range(i+1, k):
        diag = [0]*N
        for idx in range(N):
            if idx & (2**i) and idx & (2**j):
                diag[idx] = 2
        cz = Cliff(diag)
        assert (cz*cz) == Cliff([0]*N)
        gens.append(cz)

    #G = mulclose_fast(gens, verbose=True)
    found = mulclose_find(gens, target, verbose=True)
    print("found:", found)

    

if __name__ == "__main__":

    test_diag()
    test_cliff()
    test_target()


    print("OK")


