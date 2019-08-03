#!/usr/bin/env python3

import numpy
import random

from qupy.ldpc.solve import pushout, array2, parse, dot2, compose2, eq2, zeros2
from qupy.ldpc.solve import rand2, find_kernel, image
from qupy.ldpc import solve
from qupy.ldpc import css
from qupy.ldpc.chain import Chain, Map
from qupy.ldpc import chain
from qupy.argv import argv


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
    for v in image(H.transpose()):
        v = (u+v)%2
        d = min(d, v.sum()) or d
        if v.sum()==4:
            print(v)
    print("distance:", d)


def test_selfdual():

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


def test_colimit():

    n = 4
    m = n-1
    H = zeros2(m, n)
    for i in range(m):
        H[i, i] = 1
        H[i, i+1] = 1

    A = Chain([H])
    B = Chain([zeros2(0, 0)])
    C = Chain([array2([[1]])])

    CAm = zeros2(m, 1)
    CAm[0, 0] = 1
    CAm[m-1, 0] = 1
    CAn = zeros2(n, 1)
    CAn[0, 0] = 1
    CAn[n-1, 0] = 1
    CA = Map(C, A, [CAm, CAn])

    CBm = zeros2(0, 1)
    CBn = zeros2(0, 1)
    CB = Map(C, B, [CBm, CBn])

    AD, BD, D = chain.pushout(CA, CB)
    assert eq2(D[0], array2([[1,1,1], [0,1,1]])) # glue two checks at a bit

    # --------------------------

    A = Chain([H.transpose()])
    B = Chain([zeros2(0, 0)])
    C = Chain([zeros2(1, 0)])

    CAn = zeros2(n, 1)
    CAn[0, 0] = 1
    CAn[n-1, 0] = 1
    CAm = zeros2(m, 0)
    CA = Map(C, A, [CAn, CAm])

    CBn = zeros2(0, 1)
    CBm = zeros2(0, 0)
    CB = Map(C, B, [CBn, CBm])

    AD, BD, D = chain.pushout(CA, CB)
    D = D[0]
    #print(D)
    assert eq2(D, array2([[1,0,1], [1,1,0], [0,1,1]])) # glue two bits


def test_equalizer():

    n = 4
    m = n-1
    H = zeros2(m, n)
    for i in range(m):
        H[i, i] = 1
        H[i, i+1] = 1

    A = Chain([H])
    C = Chain([array2([[1]])])

    fm = zeros2(m, 1)
    fm[m-1, 0] = 1
    fn = zeros2(n, 1)
    fn[n-1, 0] = 1
    f = Map(C, A, [fm, fn])

    gm = zeros2(m, 1)
    gm[0, 0] = 1
    gn = zeros2(n, 1)
    gn[0, 0] = 1
    g = Map(C, A, [gm, gn])

    AD, BD, D = chain.equalizer(f, g)
    assert eq2(D[0], array2([[1,1,1], [0,1,1]])) # glue two checks at a bit

    # --------------------------

    A = Chain([H.transpose()])
    C = Chain([zeros2(1, 0)])

    fn = zeros2(n, 1)
    fn[0, 0] = 1
    fm = zeros2(m, 0)
    f = Map(C, A, [fn, fm])

    gn = zeros2(n, 1)
    gn[n-1, 0] = 1
    gm = zeros2(m, 0)
    g = Map(C, A, [gn, gm])

    AD, BD, D = chain.equalizer(f, g)
    D = D[0]
    #print(D)
    assert eq2(D, array2([[1,0,1], [1,1,0], [0,1,1]])) # glue two bits


def wenum(H):
    m, n = H.shape
    K = find_kernel(H)
    #print(H)
    #print(K)
    w = dict((i, []) for i in range(n+1))
    for v in image(K.transpose()):
        assert dot2(H, v).sum() == 0
        w[v.sum()].append(v)
    w = [w[i] for i in range(n+1)]
    return w


def glue2(H1, H2, i1, i2):

    m1, n1 = H1.shape
    m2, n2 = H2.shape

    A1 = Chain([H1])
    A2 = Chain([H2])
    C  = Chain([array2([[1]])])

    C1n = zeros2(n1, 1)
    C1n[i1, 0] = 1
    C1m = dot2(H1, C1n)
    C1 = Map(C, A1, [C1m, C1n])

    C2n = zeros2(n2, 1)
    C2n[i2, 0] = 1
    C2m = dot2(H2, C2n)
    C2 = Map(C, A2, [C2m, C2n])

    AD, BD, D = chain.pushout(C1, C2)

    H = D[0]
    #print(H.shape)
    #print(H)
    return H


def glue1(H, i1, i2):

    m, n = H.shape

    A = Chain([H])
    C  = Chain([array2([[1]])])

    fn = zeros2(n, 1)
    fn[i1, 0] = 1
    fm = dot2(H, fn)
    f = Map(C, A, [fm, fn])

    gn = zeros2(n, 1)
    gn[i2, 0] = 1
    gm = dot2(H, gn)
    g = Map(C, A, [gm, gn])

    _, _, D = chain.equalizer(f, g)

    H = D[0]
    #print(H.shape)
    #print(H)
    return H



def test_glue():

    m, n = 7, 10
    m = argv.get("m", 7)
    n = argv.get("n", 10)

    H = rand2(m, n)
    print(H)
    w = wenum(H)
    print("wenum:", [len(wi) for wi in w])
    v = w[4][0]
    print("v:", v)

    i0 = argv.i0
    i1 = argv.i1
    H1 = glue1(H, i0, i1)
    print(H1)

    w = wenum(H1)
    print([len(wi) for wi in w])
    #for v in w[4]:
    #    print(v)




    


if __name__ == "__main__":

    _seed = argv.get("seed")
    if _seed is not None:
        numpy.random.seed(_seed)
        random.seed(_seed)

    #test()
    #test_selfdual()
    test_colimit()
    test_equalizer()
    test_glue()

    print("OK")


