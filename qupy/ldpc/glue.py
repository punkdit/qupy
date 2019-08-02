#!/usr/bin/env python3

from qupy.ldpc.solve import pushout, array2, parse, dot2, compose2, eq2
from qupy.ldpc.css import CSSCode


def test():

    HxA = array2([[1,1,1,0,0],[0,0,1,1,1]])
    HzA = array2([[1,0,1,1,0],[0,1,1,0,1]])
    assert dot2(HxA, HzA.transpose()).sum() == 0

    HxB = array2([[1,1,1,0,0],[0,0,1,1,1]])
    HzB = array2([[1,0,1,1,0],[0,1,1,0,1]])
    assert dot2(HxB, HzB.transpose()).sum() == 0

    HzC = array2(shape=(0, 2))
    HxC = array2([[1,1]])
    assert dot2(HxC, HzC.transpose()).sum() == 0

    #code = CSSCode(Hx=Hx, Hz=Hz)
    #print(code)

    # Chain map from C -> A
    CAz = array2(shape=(2, 0))
    CAn = array2([[0,0],[0,0],[0,0],[1,0],[0,1]])
    CAx = array2([[0],[1]])

    compose2(HzC.transpose(), CAn)
    compose2(CAz, HzA.transpose())
    assert eq2(compose2(HzC.transpose(), CAn), compose2(CAz, HzA.transpose()))
    assert eq2(compose2(HxC, CAx), compose2(CAn, HxA))

    # Chain map from C -> B
    CBz = array2(shape=(2,0))
    CBn = array2([[1,0],[0,1],[0,0],[0,0],[0,0]])
    CBx = array2([[1],[0]])

    compose2(HzC.transpose(), CBn)
    compose2(CBz, HzB.transpose())
    assert eq2(compose2(HzC.transpose(), CBn), compose2(CBz, HzB.transpose()))
    assert eq2(compose2(HxC, CBx), compose2(CBn, HxB))

    Az, Bz = pushout(CAz, CBz)

    print(Az)
    print(Bz)

    An, Bn = pushout(CAn, CBn)

    print(An)
    print(Bn)

    Ax, Bx = pushout(CAx, CBx)

    print(Ax.shape)
    print(Bx.shape)

    _, _, Hzt = pushout(CAz, CBz,
        compose2(HzA.transpose(), An),
        compose2(HzB.transpose(), Bn))

    _, _, Hx = pushout(CAn, CBn,
        compose2(HxA, Ax),
        compose2(HxB, Bx))

    code = CSSCode(Hx=Hx, Hz=Hzt.transpose())
    print(code)



if __name__ == "__main__":
    test()



