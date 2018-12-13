#!/usr/bin/env python3

import sys

import numpy


from qupy.tool import write



def shortstr(A, lineno=False):
    if type(A) is list:
        A = numpy.array(A)
    A = A.view()
    assert len(A.shape)<=2
    if 1 in A.shape:
        A.shape = (1, A.shape[0]*A.shape[1])
    if len(A.shape)==1:
        A.shape = (1, A.shape[0])
    m, n = A.shape
    rows = []
    for idx in range(m):
        row = list(A[idx, :])
        row = ''.join(str(int(i)) for i in row)
        row = row.replace('0', '.')
        if lineno:
            row = "%3d %s"%(idx, row)
        rows.append(row)
    return '\n'.join(rows)


def save(H, name):
    f = open(name, "w")

    m, n = H.shape
    for row in range(m):
        row = list(H[row, :])
        row = ''.join(str(i) for i in row)
        row = row.replace('0', '.')
        print(row, file=f)
    f.close()


def load(name):
    f = open(name)
    rows = []
    for line in f:
        line = line.strip()
        line = line.replace('.', '0')
        row = [int(c) for c in line]
        rows.append(row)
    H = numpy.array(rows, dtype=numpy.int32)
    return H


def load_alist(name):
    f = open(name)

    line = f.next() # cols, rows
    line = line.strip()
    flds = line.split()
    n, m = int(flds[0]), int(flds[1]) # 20000, 10000
    
    line = f.next() # j, k (3, 6)
    flds = line.split()
    j, k = int(flds[0]), int(flds[1])

    line = f.next() # column weights (3...)
    line = f.next() # row weights (6...)

    H = numpy.zeros((m, n), dtype=numpy.int32)
    for col in range(n):
        line = f.next()
        line = line.strip()
        rows = line.split()
        assert len(rows) == j
        for row in rows:
            if row == '0':
                break
            H[int(row)-1, col] = 1

    for row in range(m):
        line = f.next()
        line = line.strip()
        cols = line.split()
        assert len(cols) == k
        for col in cols:
            if col == '0':
                break
            assert H[row, int(col)-1] == 1

    return H


def save_alist(name, H, j=None, k=None):

    if j is None:
        # column weight
        j = H[:, 0].sum()

    if k is None:
        # row weight
        k = H[0, :].sum()

    m, n = H.shape # rows, cols
    f = open(name, 'w')
    print(n, m, file=f)
    print(j, k, file=f)

    for col in range(n):
        print( H[:, col].sum(), end=" ", file=f)
    print(file=f)
    for row in range(m):
        print( H[row, :].sum(), end=" ", file=f)
    print(file=f)

    for col in range(n):
        for row in range(m):
            if H[row, col]:
                print( row+1, end=" ", file=f)
        print(file=f)

    for row in range(m):
        for col in range(n):
            if H[row, col]:
                print(col+1, end=" ", file=f)
        print(file=f)
    f.close()



def convert_toalist():
    H = load(argv.load)
    j = argv.get('j')
    k = argv.get('k')
    save_alist(argv.alist, H, j, k)
    



