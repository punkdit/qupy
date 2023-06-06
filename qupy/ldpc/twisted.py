#!/usr/bin/env python3
"""
the twisted Bacon Shor code has distance 1.
"""

import numpy

from qupy.argv import argv
from qupy.ldpc.css import CSSCode
from qupy.ldpc.solve import dot2, array2, shortstr, zeros2, rank, linear_independent



n = 13

def lookup(row, col):
    return (col + 8*row)%n

xop = set()
zop = set()
for row in range(5):
  for col in range(5):
    a = lookup(row, col)
    b = lookup(row+1, col)
    a, b = (b,a) if a>b else (a,b)
    xop.add((a,b))

    b = lookup(row, col+1)
    a, b = (b,a) if a>b else (a,b)
    zop.add((a,b))

print(xop)
print(zop)

Gx = zeros2(len(xop), n)
Gz = zeros2(len(zop), n)

Gx = []
for (a,b) in xop:
    v = zeros2(n)
    v[a] = 1
    v[b] = 1
    Gx.append(v)
Gx = array2(Gx)
Gx = linear_independent(Gx)
assert rank(Gx) == len(Gx)

Gz = []
for (a,b) in zop:
    v = zeros2(n)
    v[a] = 1
    v[b] = 1
    Gz.append(v)
Gz = array2(Gz)
Gz = linear_independent(Gz)
assert rank(Gz) == len(Gz)

print(shortstr(Gx))

code = CSSCode(Gx=Gx, Gz=Gz)
print(code)
print(code.Lx)

l = code.Lx[0]
gx = len(Gx)
d = n
best_l = l
for bits in numpy.ndindex((2,)*gx):
    v = array2(bits)
    v = dot2(v, Gx)
    v = (v+l)%2
    if v.sum() < d:
        d = v.sum()
        best_l = v

print(best_l, d)

print("OK\n")


