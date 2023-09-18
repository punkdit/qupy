#!/usr/bin/env python3

from random import randint, seed, choice, shuffle

import numpy
import numpy.random as ra
from numpy.linalg import lstsq

from qupy.ldpc.css import CSSCode, randcss
from qupy.ldpc.solve import shortstr, zeros2, array2, dot2, parse, linear_independent, solve, rank
from qupy.tool import write, choose
from qupy.argv import argv

from qupy.ldpc.bpdecode import RadfordBPDecoder
from qupy.ldpc.cluster import ClusterCSSDecoder




def bigger(H, weight=6):
    m, n = H.shape
    H1 = H.copy()
    for i in range(m):
        row = H[i]
        while 1:
            j = randint(0, m-1)
            if j==i:
                continue
            sow = H[j]
            tow = (row+sow)%2
            if tow.sum()==weight:
                break
        H1[i] = tow
    H1 = linear_independent(H1)
    #print(len(H1), m)
    while len(H1) < m:
        rows = list(H1)
        row = H[randint(0, m-1)]
        rows.append(row)
        H1 = array2(rows)
        H1 = linear_independent(H1)
    #print(len(H1))
    return H1


def make_bigger(H, weight=6): # why is this slower ?!?
    m, n = H.shape
    rows = []
    for i in range(m):
      for j in range(i+1, m):
        u = (H[i]+H[j])%2
        if u.sum() == weight:
            rows.append(u)
    #print("rows:", len(rows))
    #R = array2(rows)
    #print(rank(R), m)
    while 1:
        shuffle(rows)
        #H1 = [choice(rows) for i in range(m)]
        H1 = rows #[:m]
        H1 = array2(H1)
        H1 = linear_independent(H1)
        while len(H1)<m:
            u = H[randint(0, m-1)]
            v = solve(H1.transpose(), u)
            if v is None:
                u.shape = (1, n)
                H1 = numpy.concatenate((H1, u))
                #print(len(H1), m)
        H1 = array2(H1)
        assert rank(H1) == m
        yield H1
        


def reweight_slow(code):
    while 1:
        Hx = bigger(code.Hx)
        Hz = bigger(code.Hz)
        yield CSSCode(Hx=Hx, Hz=Hz, Lx=code.Lx, Lz=code.Lz, build=False)
        write("/")

def reweight(code):
    for (Hx, Hz) in zip(make_bigger(code.Hx), make_bigger(code.Hz)):
        yield CSSCode(Hx=Hx, Hz=Hz, Lx=code.Lx, Lz=code.Lz, build=False)
        write("/")


def main():

    from qupy.ldpc.toric import Toric2D
    
    l = argv.get('l', 8)
    li = argv.get('li', l)
    lj = argv.get('lj', l)
    si = argv.get('si', 0)
    sj = argv.get('sj', 0)

    toric = Toric2D(li, lj, si, sj)
    Hx, Hz = toric.Hx, toric.Hz
    strop = toric.strop
    #print("Hx:")
    #print(shortstr(Hx))
    #print("Hz:")
    #print(shortstr(Hz))
    #print("Lx:")
    #print(shortstr(toric.Lx))
    code = CSSCode(Hx=Hx, Hz=Hz, Lx=toric.Lx, Lz=toric.Lz)

    dup = argv.get("dup", 1)
    print("building...")
    codes = reweight(code)
    codes = [codes.__next__() for i in range(dup)]
    print("done")

    N = argv.get('N', 10)
    p = argv.get('p', 0.04)
    if argv.silent:
        global write
        write = lambda s:None

    decoders = [RadfordBPDecoder(2, code.Hz) for code in codes]

    if argv.noerr:
        print("redirecting stderr to stderr.out")
        fd = os.open("stderr.out", os.O_CREAT|os.O_WRONLY)
        os.dup2(fd, 2)

    distance = code.n
    count = 0
    failcount = 0
    nonuniq = 0

    scramble = lambda err_op, H:(err_op + dot2(ra.binomial(1, 0.5, (len(H),)), H)) % 2

    for i in range(N):

        # We use Hz to look at X type errors (bitflip errors)
        err_op = ra.binomial(1, p, (code.n,))

        write(str(err_op.sum()))

        s = dot2(code.Hz, err_op)
        write(":s%d:"%s.sum())

        _err_op = scramble(err_op, code.Hx)
        _err_op = scramble(_err_op, code.Lx)
        #print(_err_op)
        #_err_op = err_op
        for decoder in decoders:
            op = decoder.decode(p, _err_op, verbose=False, argv=argv)
            if op is not None:
                break

        c = 'F'
        success = False
        if op is not None:
            op = (op+err_op)%2
            # Should be a codeword of Hz (kernel of Hz)
            if dot2(code.Hz, op).sum() != 0:
                print(dot2(code.Hz, op))
                print("\n!!!!!  BUGBUG  !!!!!", sparsestr(err_op))
                continue
            write("%d:"%op.sum())

            # Are we in the image of Hx ? If so, then success.
            success = dot2(code.Lz, op).sum()==0

            if success and op.sum():
                nonuniq += 1
            #    print "\n", shortstr(err_op)
            #    return

            c = '.' if success else 'x'

            if op.sum() and not success:
                distance = min(distance, op.sum())

        else:
            failcount += 1

        write(c+' ')
        count += success

    if N:
        print()
        print(datestr)
        print(argv)
        print("error rate = %.8f" % (1. - 1.*count / (i+1)))
        print("fail rate  = %.8f" % (1.*failcount / (i+1)))
        print("nonuniq = %d" % nonuniq)
        print("distance <= %d" % distance)

    
if __name__ == "__main__":

    import os
    datestr = os.popen('date "+%F %H:%M:%S"').read().strip()

    from time import time
    start_time = time()

    _seed = argv.get('seed')
    if _seed is not None:
        seed(_seed)
        ra.seed(_seed)

    if argv.profile:
        import cProfile as profile
        profile.run("main()")

    else:
        main()

    print("%.3f seconds"%(time() - start_time))


