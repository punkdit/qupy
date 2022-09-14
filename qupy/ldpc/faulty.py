#!/usr/bin/env python

"""
fault tolerant error correction
"""


from random import randint, seed

import numpy
import numpy.random as ra
from numpy.linalg import lstsq

from qupy.ldpc.css import CSSCode, randcss
from qupy.ldpc.mps import TensorNetwork
from qupy.ldpc.solve import shortstr, zeros2, array2, dot2, parse
from qupy.tool import write, choose
from qupy.argv import argv
from qupy.smap import SMap



def run_trial(Hz, p, q, M):

    m, n = Hz.shape

    smap = SMap()
    acc_err = zeros2(n)
    
    stacks = []
    for row in range(M):
        bit_err = ra.binomial(1, p, (n,))
        acc_err = (acc_err + bit_err) % 2
        syn = dot2(Hz, acc_err)
        syn_err = ra.binomial(1, q, (m,))
        tot = (syn + syn_err) % 2
        for i in range(m):
            smap[1+row, 2*i+1] = '|' if tot[i] else '.'
        stacks.append(tot)
        for i in range(n):
            if bit_err[i]:
                smap[row, 2*i] = '_'
            elif acc_err[i]:
                smap[0+row, 2*i] = '*'
                smap[1+row, 2*i] = '*'

    print()
    print(smap)
    stacks = array2(stacks)
    #print(shortstr(stacks))
    
    net = TensorNetwork()



def main():

    n = argv.get("n", 48)
    p = argv.get("p", 0.12)
    q = argv.get("q", 0.04)
    N = argv.get("N", 1)
    M = argv.get("M", 100)

    Hz = zeros2(n-1, n)
    for i in range(n-1):
        Hz[i, i] = 1
        Hz[i, i+1] = 1
    #print(shortstr(Hz))

    for trial in range(N):
        run_trial(Hz, p, q, M)

if __name__ == "__main__":

    main()


