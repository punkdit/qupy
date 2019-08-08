#!/usr/bin/env python3

import numpy

from qupy.ldpc.solve import parse, zeros2, span, dot2, row_reduce, array2
from qupy.ldpc.css import CSSCode


def z_weld(acode, bcode, pairs):

    for c in [acode, bcode]:
        print("Lx:")
        print(c.Lx)
        print("Lz:")
        print(c.Lz)
        print("Hx:")
        print(c.Hx)
        print("Hz:")
        print(c.Hz)
        print("-------------------")

    mx = acode.mx + bcode.mx
    #for (i, j) in pairs:
    assert len(set(pairs)) == len(pairs) # uniq

    n = acode.n + bcode.n - len(pairs)

#    Hx = zeros2(mx, n)
#    Hx[:acode.mx, :acode.n] = acode.Hx
#    Hx[acode.mx:, acode.n-len(pairs):] = bcode.Hx
#        
#    az = acode.mz + bcode.mz + acode.k + bcode.k
#    Az = zeros2(az, n)
#    r0, r1 = 0, acode.mz
#    Az[r0:r1, :acode.n] = acode.Hx; r0, r1 = r1, r1+len(acode.Hx)
#    Az[r0:r1, acode.n-len(pairs):] = bcode.Hx; r0, r1 = r1, r1+len(bcode.Hx)
##    Az[r0:r1, :acode.n] = acode.Lz; r0, r1 = r1, r1+len(acode.Lz)
##    #assert r1 == len(Az), (r1, len(Az))
##    Az[r0:r1, acode.n-len(pairs):] = bcode.Lz; r0, r1 = r1, r1+len(bcode.Lz)
#
#    print("Az:")
#    print(Az)

    #print(Az)
    Hz = []
    for z in span(Az):
        #print(z)
        #print( dot2(Hx, z.transpose()))
        if dot2(Hx, z.transpose()).sum() ==0:
            Hz.append(z)
    Hz = array2(Hz)
    Hz = row_reduce(Hz)
    
    print("Hx:")
    print(Hx)
    print("Hz:")
    print(Hz)


def test():

    Hx = parse("1.11. .11.1")
    Hz = parse("111.. ..111")

    code = CSSCode(Hx=Hx, Hz=Hz)

    print(code)

    z_weld(code, code, [(3, 0), (4, 1)])



if __name__ == "__main__":
    test()


