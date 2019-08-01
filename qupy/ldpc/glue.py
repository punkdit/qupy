#!/usr/bin/env python3

from qupy.ldpc.solve import pushout, array2, parse, dot2, compose2
from qupy.ldpc.css import CSSCode


def test():

    HxA = array2([[1,1,1,0,0],[0,0,1,1,1]])
    HzA = array2([[1,0,1,1,0],[0,1,1,0,1]])
    assert dot2(HxA, HzA.transpose()).sum() == 0

    HxB = array2([[1,1,1,0,0],[0,0,1,1,1]])
    HzB = array2([[1,0,1,1,0],[0,1,1,0,1]])
    assert dot2(HxB, HzB.transpose()).sum() == 0

    #code = CSSCode(Hx=Hx, Hz=Hz)
    #print(code)

    czB = array2([[1],[0]])
    cnB = array2([[1,0],[0,0],[0,0],[0,1],[0,0]])
    cxB = array2([[], []])

    assert cxB.shape == (2, 0)

    czA = array2([[0],[1]])
    cnA = array2([[0,0],[1,0],[0,0],[0,0],[0,1]])
    cxA = array2([[], []])

    Bz, Az = pushout(czB, czA)

    print(Bz)
    print(Az)

    Bn, An = pushout(cnB, cnA)

    print(Bn)
    print(An)

    _, _, Hzt = pushout(czB, czA, 
        compose2(HzB.transpose(), Bn),
        compose2(HzA.transpose(), An)
    )




if __name__ == "__main__":
    test()



