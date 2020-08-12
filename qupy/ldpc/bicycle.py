#!/usr/bin/env python3

import sys, os
from math import *
from random import *
import time

import numpy
import numpy.random as ra


from qupy.ldpc.tool import shortstr, write


class Bicycle_LDPC(object):
    def __init__(self, m, n, j, k):
        """
        m : constraints/2 (rows)
        n : bits (cols)
        j : column weight
        k : row weight (constraint weight)
        """

        C = numpy.zeros((n//2, n//2), dtype=numpy.int32)

        # row-weight k//2
        write("building bicycle")

        cols = list(range(n//2))
        for i in range(k//2):
            col = choice(cols)
            C[0, col] = 1
            cols.remove(col)

        for i in range(1, n//2):
            C[i, i:] = C[0, :n//2-i]
            C[i, :i] = C[0, n//2-i:]

        H = numpy.zeros((n//2, n), dtype=numpy.int32)
        H[:, :n//2] = C
        H[:, n//2:] = C.transpose()

        #print shortstr(H)

        rows = list(range(n//2))

        if m is None:
            m = n*j//k
        #print m, n
#        assert k*m == n*j

        colweights = [H[:, col].sum() for col in range(n)]
        w1 = max(colweights)
        rowcache = list(rows)
        #print colweights
        while len(rows) > m:
            assert rowcache # ouch!
            row = choice(rowcache)
            w1 = max(colweights)
            w0 = min(colweights)
            for col in range(n):
                w = colweights[col]
                if w<j//2 and H[row, col]:
                #if w<w1-2 and H[row, col]: # works for m=1420, n=3786, k=12
                    rowcache.remove(row)
                    break
            else:
                for col in range(n):
                    w = colweights[col]
                    if H[row, col]:
                        colweights[col] -= 1
                        assert colweights[col] > 0
                rows.remove(row)
                rowcache.remove(row)
                write('.')
                #write("%s,%s"%(w0, w1))

        print(min(colweights), max(colweights))

        H = H[rows, :]

        #self.Hx = self.Hz = H
        self.H = H

        print("done")
            

