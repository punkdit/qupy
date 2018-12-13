#!/usr/bin/env python3

#from math import sqrt
#from random import choice, randint, seed, shuffle
#from functools import reduce
#from operator import mul, matmul, add

import numpy
from numpy import exp, pi

from qupy import scalar
#scalar.use_reals()
from qupy.scalar import EPSILON, MAX_GATE_SIZE
from qupy.dense import Qu
from qupy.util import mulclose, show_spec
from qupy.argv import argv

from qupy.dev.comm import Poly


def cyclotomic(n):
    return numpy.exp(2*numpy.pi*1.j/n)


def build():

    global Oct, Tetra, Icosa, Sym

    
    gen = [
        [[0, 1], [1, 0]],
    ]
    gen = [Qu((2, 2), 'ud', v) for v in gen]

    Sym = mulclose(gen)
    assert len(Sym)==2

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

    Oct = mulclose(gen)
    assert len(Oct)==48

    # ----------------------------------
    # binary tetrahedral group ... hacked

    i = cyclotomic(4)

    gen = [
        [[(-1+i)/2, (-1+i)/2], [(1+i)/2, (-1-i)/2]],
        [[0,i], [-i,0]]
    ]

    gen = [Qu((2, 2), 'ud', v) for v in gen]

    Tetra = mulclose(gen)
    assert len(Tetra)==48 # whoops it must be another Oct ... 
    
    Tetra = [g for g in Tetra if g in Oct]  # hack this
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


def main():

    I = Poly.identity(2)
    zero = Poly.zero(2)
    x = Poly({(1, 0): 1.})
    y = Poly({(0, 1): 1.})

    #print( (x+I) ** 3 )

    def act(g, p):
        #print("act", g, p)
        #print(g[0, 0])
        x_1 = g[0, 0]*x + g[1, 0]*y
        x_2 = g[0, 1]*x + g[1, 1]*y
        s = str(p)
        p = eval(s, {}, locals())
        return p

    def average(G, p):
        p1 = zero
        for g in G:
            p1 = p1 + act(g, p)
        return p1

    assert average(Sym, x*y**2) == x*y**2 + x**2*y

    #p = average(Tetra, x**8)
    #print(p)

    G = eval(argv.get("G", "Tetra"))

    degree = argv.get("degree", 8)
    for degree in range(1, degree+1):
        #p = Poly.random(2, degree)
        #p = (x+y+I)**8
        #print(p)
        p = x**degree
        q = average(G, p)
        if q.degree > 0:
            print("degree:", q.degree)
            print(q)


if __name__ == "__main__":

    name = argv.next() or "main"

    fn = eval(name)
    fn()


