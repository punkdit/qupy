#!/usr/bin/env python3

"""
Various 2-dimensional representations of binary
platonic groups.
"""

from math import sqrt
from random import randint, choice, seed
from functools import reduce
from operator import matmul, mul, add

#from scipy.spatial import KDTree, cKDTree
import numpy

from qupy.dense import Qu
from qupy.util import mulclose
from qupy.tool import cross, write
from qupy.argv import argv
#from qupy.dev.su2 import Icosa

EPSILON=1e-8


# See:
# https://uu.diva-portal.org/smash/get/diva2:1184051/FULLTEXT01.pdf


def cyclotomic(n):
    return numpy.exp(2*numpy.pi*1.j/n)


def build_tetra(idx=0):
    " binary _tetrahedral group "

    i = cyclotomic(4)
    r2 = sqrt(2)
    r3 = sqrt(3)
    r6 = sqrt(6)

    gen = [
        [ # same as Natural repr ?
        [[(-1+i)/2, (-1+i)/2], [(1+i)/2, (-1-i)/2]],        # order 3
        [[-1/2-1/2*i, -1/2-1/2*i], [1/2-1/2*i, -1/2+1/2*i]] # order 3
        ],
        [ # Natural repr
        [[(1+i*r3)/2, 0], [0, (1-i*r3)/2]],
        [[(r3-i)/(2*r3), (-i*2*r2)/(2*r3)], [(-i*2*r2)/(2*r3), (r3+i)/(2*r3)]],
        ],
        [ # 2A
        [[(-2)/2., 0], [0, (1+i*r3)/2.]],
        [[(2*i)/(2*r3), (r6+i*r2)/(2*r3)], [(r6+i*r2)/(2*r3), (i-r3)/(2*r3)]],
        ],
        [ # 2B
        [[(1-i*r3)/2., 0], [0, (-2)/2.]],
        [[(-r3-i)/(2*r3), (-r6+i*r2)/(2*r3)], [(-r6+i*r2)/(2*r3), (-2*i)/(2*r3)]],
        ],
    ][idx]

    gen = [Qu((2, 2), 'ud', v) for v in gen] 
    a, b = gen
    #print( len(mulclose([b], maxsize=32)))
    assert len(mulclose([a], maxsize=32)) in (3, 6)
    assert len(mulclose([b], maxsize=32)) in (3, 6)

    Tetra = mulclose(gen, maxsize=32)
    assert len(Tetra)==24

    return gen, Tetra


def build_octa(idx=0):
    " binary _octahedral group "

    x = cyclotomic(8)

    r2 = x-x**3 # sqrt(2)
    i = x**2 

    gen = [
        [
        [[(-1+i)/2,(-1+i)/2], [(1+i)/2,(-1-i)/2]], 
        [[(1+i)/r2,0], [0,(1-i)/r2]]
        ],
        [ # 2A
        [[(1.+i)/r2, 0.], [0., (1-i)/r2]],
        [[1./r2, -i/r2], [-i/r2, 1./r2]],
        ],
        [ # 2B
        [[(-1.-i)/r2, 0.], [0., (-1.+i)/r2]],
        [[-1./r2, i/r2], [i/r2, -1./r2]],
        ],
#        [ # 2C
#        [[1., 0.], [-1., -1.]],
#        [[0., 1.], [1., 0.]],
#        ],
    ][idx]
    gen = [Qu((2, 2), 'ud', v) for v in gen] 
    octa_gen = gen
    #a, b = gen
    #print((a**3*b).trace())

    Octa = mulclose(gen)
    assert len(Octa)==48, len(Octa)
    #print("Octa:", Octa.words.values())

    return gen, Octa


def build_icosa(idx=0):
    " binary _icosahedral group "

    i = cyclotomic(4)
    v = cyclotomic(10)
    z = v**2 
    r5 = 2*z**3 + 2*z**2 + 1 #sqrt(5)
    gen = [
        [ # Same as Nat ?
        [[z**3, 0], [0, z**2]], 
        [[(z**4-z)/r5, (z**2-z**3)/r5], 
        [(z**2-z**3)/r5, -(z**4-z)/r5]]
        ],
        [ # Nat
        [[v, 0], [0, v**9]],
        [[(5*(v-v**4)-r5*(v+v**4))/10., (-2*r5*(v+v**4))/10.],
        [(-2*r5*(v+v**4))/10., (5*(v-v**4)+r5*(v+v**4))/10.]]
        ],
        [
        [[0, i], [i, (1-r5)/2]],
        [[-i, i], [-(1-r5)/2., (1-r5)/2+i]],
        ]
    ][idx]
    
    gen = [Qu((2, 2), 'ud', v) for v in gen] 
    #a, b = gen
    #H = mulclose([b], maxsize=100)
    #print(len(H))

    X = Qu((2, 2), 'ud', [[0., 1], [1., 0.]])
    Z = Qu((2, 2), 'ud', [[1., 0], [0., -1.]])

    G = mulclose(gen, maxsize=256)

    assert len(G)==120

    if idx in (0,1):
        assert X*Z in G
        assert X not in G
        assert Z not in G

    return gen, G


def build_pauli(*args):

    X = Qu((2, 2), 'ud', [[0., 1], [1., 0.]])
    Z = Qu((2, 2), 'ud', [[1., 0], [0., -1.]])
    gen = [X, Z]

    G = mulclose(gen)

    assert len(G)==8

    return gen, G


def build(name, *args):
    G = None
    if name == "icosa":
        gen, G = build_icosa(*args)
    elif name == "tetra":
        gen, G = build_tetra(*args)
    elif name == "octa":
        gen, G = build_octa(*args)
    elif name == "pauli":
        gen, G = build_pauli(*args)

    return G




