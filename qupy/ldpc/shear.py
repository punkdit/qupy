#!/usr/bin/env python3



import numpy
import numpy.random as ra

from qupy.argv import argv
from qupy.tool import write
from qupy.ldpc.solve import dot2, shortstr
from qupy.ldpc.toric import Toric2D
from qupy.ldpc.css import CSSCode
from qupy.ldpc.cluster import ClusterCSSDecoder

if __name__ == "__main__":
    import os
    datestr = os.popen('date "+%F %H:%M:%S"').read().strip()




def main():


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
    #print("Lx:")
    #print(shortstr(toric.Lx))
    code = CSSCode(Hx=Hx, Hz=Hz, Lx=toric.Lx, Lz=toric.Lz)

    print(code)
    print(code.Lx)

    decoder = ClusterCSSDecoder(2, code.Hx, code.Hz)

    N = argv.get("N", 0)
    p = argv.get("p", 0.05)
    verbose = argv.verbose

    distance = code.n
    count = 0
    failcount = 0
    nonuniq = 0

    total = numpy.array([0.]*code.k)

    for trial in range(N):

        err_op = ra.binomial(1, p, (code.n,))
        err_op = err_op.astype(numpy.int32)

        write(str(err_op.sum()))
        #print err_op.shape

        s = dot2(code.Hz, err_op)
        write(":s%d:"%s.sum())

        op = decoder.decode(p, err_op, verbose=verbose, argv=argv)

        c = 'F'
        success = False
        result = numpy.array([1]*code.k)
        if op is not None:
            op = (op+err_op)%2
            # Should be a codeword of Hz (kernel of Hz)
            if dot2(code.Hz, op).sum() != 0:
                print(dot2(code.Hz, op))
                print("\n!!!!!  BUGBUG  !!!!!", sparsestr(err_op))
                continue
            write("%d:"%op.sum())

            # Are we in the image of Hx ? If so, then success.
            result = dot2(code.Lz, op)
            success = result.sum()==0

            #print(result, end=" ")

            if success and op.sum():
                nonuniq += 1
            #    print "\n", shortstr(err_op)
            #    return

            c = '.' if success else 'x'

            if op.sum() and not success:
                distance = min(distance, op.sum())


        else:
            failcount += 1
 
            if failures:
                print(shortstr(err_op), file=failures)
                failures.flush()

        total += result
 
        write(c+' ')
        count += success

    if N:
        print()
        print(datestr)
        print(argv)
        print("error rate = %.8f" % (1. - 1.*count / (trial+1)))
        print("fail rate = %.8f" % (1.*failcount / (trial+1)))
        print("nonuniq = %d" % nonuniq)
        print("distance <= %d" % distance)
        total = total/(trial+1)
        print("logop errors = [%s]" % ', '.join(str(x) for x in total))



if __name__ == "__main__":

    main()


