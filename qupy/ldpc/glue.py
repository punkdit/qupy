#!/usr/bin/env python3

import numpy

from qupy.ldpc.solve import pushout, array2, parse, dot2, compose2, eq2, zeros2
from qupy.ldpc import solve
from qupy.ldpc import css
from qupy.ldpc.chain import Chain, Map
from qupy.ldpc import chain


def test():

    HxA = array2([[1,1,1,0,0],[0,0,1,1,1]])
    HzA = array2([[1,0,1,1,0],[0,1,1,0,1]])
    A = Chain([HxA, HzA.transpose()])

    HxB = array2([[1,1,1,0,0],[0,0,1,1,1]])
    HzB = array2([[1,0,1,1,0],[0,1,1,0,1]])
    B = Chain([HxB, HzB.transpose()])

    HzC = zeros2(0, 2)
    HxC = array2([[1,1]])
    C = Chain([HxC, HzC.transpose()])

    # Chain map from C -> A
    CAz = array2(shape=(2, 0))
    CAn = array2([[0,0],[0,0],[0,0],[1,0],[0,1]])
    CAx = array2([[0],[1]])
    CA = Map(C, A, [CAx, CAn, CAz])

    # Chain map from C -> B
    CBz = CAz
    CBn = CAn
    CBx = CAx
    CB = Map(C, B, [CBx, CBn, CBz])

    AD, BD, D = chain.pushout(CA, CB)
    code = D.get_code()
    assert code.mx == 3
    assert code.mz == 4
    assert code.k == 1

    #print(code.longstr())

    # Chain map from C -> D
    CDz = zeros2(4, 0)
    CDn = zeros2(8, 2)
    CDn[4,0] = 1
    CDn[3,1] = 1
    CDx = zeros2(3, 1)
    CDx[1,0] = 1
    CD = Map(C, D, [CDx, CDn, CDz])

    _, _, E = chain.pushout(CA, CD)
    code = E.get_code()
    #print(code.longstr())

    return

    #dual = code.dual()
    #print(css.lookup_distance(code))
    print("Hz:")
    print(code.Hz)
    print("Hx:")
    print(code.Hx)
    print("Lx:")
    print(code.Lx)
    print()

    #H = numpy.concatenate((code.Lx, code.Hx))
    u = code.Lx
    H = code.Hx
    #print(H)
    d = H.shape[1]
    for v in solve.image(H.transpose()):
        v = (u+v)%2
        d = min(d, v.sum()) or d
        if v.sum()==4:
            print(v)
    print("distance:", d)


def test_color():

    HxA = array2([[1,1,1,1]])
    HzA = array2([[1,1,1,1]])
    A = Chain([HxA, HzA.transpose()])

    HxB = array2([[1,1,1,1]])
    HzB = array2([[1,1,1,1]])
    B = Chain([HxB, HzB.transpose()])

    HzC = zeros2(0, 2)
    HxC = array2([[1,1]])
    C = Chain([HxC, HzC.transpose()])

    #HzC = zeros2(0, 2)
    #HxC = zeros2(0, 2)
    #C = Chain([HxC, HzC.transpose()])

    # Chain map from C -> A
    CAz = zeros2(1, 0)
    CAn = zeros2(4, 2)
    CAn[0, 0] = 1
    CAn[1, 1] = 1
    CAx = array2([[1]])
    CA = Map(C, A, [CAx, CAn, CAz])

    # Chain map from C -> B
    CBz = CAz
    CBn = CAn
    CBx = CAx
    CB = Map(C, B, [CBx, CBn, CBz])

    AD, BD, D = chain.pushout(CA, CB)
    code = D.get_code()
    #print(code)


if __name__ == "__main__":
    #test()
    test_color()



