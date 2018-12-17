#!/usr/bin/env python3

import numpy

write = lambda s : print(s, end='', flush=True)


EPSILON = 1e-8

def fstr(x):
    if abs(x.imag)<EPSILON and abs(x.real-int(round(x.real)))<EPSILON:
        x = int(round(x.real))
    elif abs(x.imag)<EPSILON:
        x = x.real
        for digits in range(4):
            rx = round(x, digits)
            if abs(x-rx)<EPSILON:
                x = rx
        
    return str(x)


def XXastr(A):
    B = numpy.zeros(A.shape, dtype=object)
    for idx in numpy.ndindex(A.shape):
        #print(idx, A[idx])
        B[idx] = fstr(A[idx])
    return repr(B)


def astr(A):
    m, n = A.shape
    rows = []
    for i in range(m):
        row = ' '.join(fstr(x) for x in A[i])
        rows.append(row)
    s = '\n'.join(rows)
    return s


def cross(itemss):
    if len(itemss)==0:
        yield ()
    else:
        for head in itemss[0]:
            for tail in cross(itemss[1:]):
                yield (head,)+tail

