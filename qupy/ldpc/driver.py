#!/usr/bin/env python3

from random import randint, seed

import numpy
import numpy.random as ra
from numpy.linalg import lstsq

from qupy.ldpc.css import CSSCode, randcss
from qupy.ldpc.solve import shortstr, zeros2, array2, dot2, parse, linear_independent
from qupy.tool import write, choose
from qupy.argv import argv

from qupy.ldpc.bpdecode import RadfordBPDecoder
from qupy.ldpc.cluster import ClusterCSSDecoder


if __name__ == "__main__":
    import os
    datestr = os.popen('date "+%F %H:%M:%S"').read().strip()


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


def reweight(code):
    Hx = bigger(code.Hx)
    Hz = bigger(code.Hz)
    return CSSCode(Hx=Hx, Hz=Hz)


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

    dup = argv.get("dup", 4)
    codes = [reweight(code) for i in range(dup)]

    N = argv.get('N', 0)
    p = argv.get('p', 0.03)
    weight = argv.weight

    decoders = [RadfordBPDecoder(2, code.Hz) for code in codes]

    if argv.noerr:
        print("redirecting stderr to stderr.out")
        fd = os.open("stderr.out", os.O_CREAT|os.O_WRONLY)
        os.dup2(fd, 2)

    distance = code.n
    count = 0
    failcount = 0
    nonuniq = 0

    for i in range(N):

        # We use Hz to look at X type errors (bitflip errors)
        err_op = ra.binomial(1, p, (code.n,))
        err_op = err_op.astype(numpy.int32)

        write(str(err_op.sum()))

        s = dot2(code.Hz, err_op)
        write(":s%d:"%s.sum())

        for decoder in decoders:
            op = decoder.decode(p, err_op, verbose=False, argv=argv)
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

            #print "is_stab:"
            #print strop(op)
            # Are we in the image of Hx ? If so, then success.
            #success = code.is_stab(op)
            success = dot2(code.Lz, op).sum()==0

            if success and op.sum():
                nonuniq += 1
            #    print "\n", shortstr(err_op)
            #    return

            c = '.' if success else 'x'

            if argv.k_rank:

                # XX this does not work very well...

                a = dot2(code.Lz, op)
                assert (a.sum()==0) == success

                if not success:
                    #r = -1. // (a.sum()**0.5) # sqrt ??
                    r = -1. / a.sum()
                    k_ranks += r*a
                else:
                    #r = 1. / (code.k**0.5)
                    r = 1. / code.k
                    k_ranks += r*a

                k_record.append((a, success))


            if op.sum() and not success:
                distance = min(distance, op.sum())

            if not success and argv.showfail:
                print()
                print("FAIL:")
                print(shortstr(err_op))

        else:
            failcount += 1

        write(c+' ')
        count += success

    if hasattr(decoder, 'fini'):
        decoder.fini()

    if N:
        print()
        print(datestr)
        print(argv)
        print("error rate = %.8f" % (1. - 1.*count / (i+1)))
        print("fail rate = %.8f" % (1.*failcount / (i+1)))
        print("nonuniq = %d" % nonuniq)
        print("distance <= %d" % distance)

    

class MultiDecoder(object):
    def __init__(self, decode0, decode1):
        self.decode0 = decode0
        self.decode1 = decode1

    def decode(self, p, err_op, **kw):
        decode0 = self.decode0
        decode1 = self.decode1
        op = decode0.decode(p, err_op, **kw)
        if op is None:
            write("F")
            op = decode1.decode(p, err_op, **kw)
        return op


if __name__ == "__main__":

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


