#!/usr/bin/env python3

from functools import reduce
from operator import mul, matmul, add
from math import sqrt

import numpy

from qupy.dense import Qu
from qupy.dev.su2 import (StabilizerCode, EPSILON, cyclotomic,
    promote_pauli, decompose)

from qupy.argv import argv

if argv.fast:
    #print("importing _algebra")
    from qupy.dev._algebra import Algebra, build_algebra, Tensor
else:
    from qupy.dev.algebra import Algebra, build_algebra, Tensor



def build_rm(pauli):
    """
Lx:Lz =
...1.111.111111. ............1111
......11..111111 .........1.11.1.
.....1.1.1.11111 ..........1111..
............1111 ...1...1.1111...
.........11..11. ......11..11....
..........11..11 .....11..11.....
Hx:Tz =
1111111111111111 1...............
.1.1.1.1.1.1.1.1 11..............
..11..11..11..11 1.1.............
....1111....1111 1...1...........
........11111111 1.......1.......
Tx:Hz =
...1.11..11.1..1 1111111111111111
......11..1111.. .1.1.1.1.1.1.1.1
.....1.1.1.11.1. ..11..11..11..11
...1...1.11..11. ....1111....1111
...1.11....1.11. ........11111111
    """

    sx = """
    .X.X.X.X.X.X.X.X
    ..XX..XX..XX..XX
    ....XXXX....XXXX
    ........XXXXXXXX
    """.replace(".", "I")

    sz = """
    .Z.Z.Z.Z.Z.Z.Z.Z
    ..ZZ..ZZ..ZZ..ZZ
    ....ZZZZ....ZZZZ
    ........ZZZZZZZZ
    ............ZZZZ
    .........Z.ZZ.Z.
    ..........ZZZZ..
    ...Z...Z.ZZZZ...
    ......ZZ..ZZ....
    .....ZZ..ZZ.....
    """.replace(".", "I")
    sz = """
    .Z.Z.Z.Z.Z.Z.Z.Z
    ..ZZ..ZZ..ZZ..ZZ
    ....ZZZZ....ZZZZ
    ........ZZZZZZZZ
    """.replace(".", "I")

    code = StabilizerCode(pauli, sx+sz)

    return code


def build_color(pauli):
    sx = """
    X.X.X.X.X.X.X.X
    .XX..XX..XX..XX
    ...XXXX....XXXX
    .......XXXXXXXX
    """.replace(".", "I")

    sz = """
    Z.Z.Z.Z.Z.Z.Z.Z
    .ZZ..ZZ..ZZ..ZZ
    ...ZZZZ....ZZZZ
    .......ZZZZZZZZ
    """.replace(".", "I")

    code = StabilizerCode(pauli, sx+sz)

    return code


def build_code(pauli, name=None):
    I = pauli.I
    X = pauli.X
    Y = pauli.Y
    Z = pauli.Z

    def mk_stab(s):
        s = s.replace(".", "I")
        sx = s.replace("1", "X")
        sz = s.replace("1", "Z")
        code = StabilizerCode(pauli, sx+sz)
        return code

    if name is None:
        name = argv.next()

    if name=="two":
        code = StabilizerCode(pauli, "XX ZZ")
    elif name=="four":
        code = StabilizerCode(pauli, "XXII ZZII IIXX IIZZ")
    elif name=="five":
        code = StabilizerCode(pauli, "XZZXI IXZZX XIXZZ ZXIXZ")
        op = (I@I@I@I@I+I@X@Z@Z@X-I@Z@Y@Y@Z-I@Y@X@X@Y
            +X@I@X@Z@Z-X@X@Y@I@Y+X@Z@Z@X@I-X@Y@I@Y@X
            -Z@I@Z@Y@Y+Z@X@I@X@Z+Z@Z@X@I@X-Z@Y@Y@Z@I
            -Y@I@Y@X@X-Y@X@X@Y@I-Y@Z@I@Z@Y-Y@Y@Z@I@Z)
        assert op == code.get_projector()
    elif name=="seven":
        code = StabilizerCode(pauli, "XZZXIII IXZZXII IIXZZXI IIIXZZX XIIIXZZ ZXIIIXZ")
    elif name=="steane":
        code = StabilizerCode(pauli, "XXXXIII XXIIXXI XIXIXIX ZZZZIII ZZIIZZI ZIZIZIZ")
    elif name=="rm":
        code = build_rm(pauli)
    elif name=="color":
        code = build_color(pauli)
    elif name=="toric":
        s = """
        XX.XX...
        X...XX.X
        .XXX..X.
        ZZZ..Z..
        Z.ZZ...Z
        .Z..ZZZ.
        """.replace(".", "I")
        code = StabilizerCode(pauli, s)
    return code



def get_projector():
    pauli = build_algebra("IXZY",
        "X*X=I Z*Z=I Y*Y=-I X*Z=Y Z*X=-Y X*Y=Z Y*X=-Z Z*Y=-X Y*Z=X")

    I = pauli.I
    X = pauli.X
    Y = pauli.Y
    Z = pauli.Z

    code = build_code(pauli)
    op = code.get_projector()
    return op


def get_gauge(pauli):

    I = pauli.I
    X = pauli.X
    Y = pauli.Y
    Z = pauli.Z

    RM24 = """
    1111111111111111
    .1.1.1.1.1.1.1.1
    ..11..11..11..11
    ....1111....1111
    ........11111111
    ...1...1...1...1
    .....1.1.....1.1
    .........1.1.1.1
    ......11......11
    ..........11..11
    ............1111
    """

    sx = """
    X.X.X.X.X.X.X.X
    .XX..XX..XX..XX
    ...XXXX....XXXX
    .......XXXXXXXX
    ..X...X...X...X
    ....X.X.....X.X
    ........X.X.X.X
    .....XX......XX
    .........XX..XX
    ...........XXXX
    """.replace(".", "I")

    sz = """
    Z.Z.Z.Z.Z.Z.Z.Z
    .ZZ..ZZ..ZZ..ZZ
    ...ZZZZ....ZZZZ
    .......ZZZZZZZZ
    ..Z...Z...Z...Z
    ....Z.Z.....Z.Z
    ........Z.Z.Z.Z
    .....ZZ......ZZ
    .........ZZ..ZZ
    ...........ZZZZ
    """.replace(".", "I")

    ops = sx+sz
    ops = ops.strip().split()
    ops = [pauli.parse(s) for s in ops]
    H = reduce(add, ops)
    return H


def main():

    _I = Qu((2, 2), 'ud', [[1, 0], [0, 1]])
    _X = Qu((2, 2), 'ud', [[0, 1], [1, 0]])
    _Z = Qu((2, 2), 'ud', [[1, 0], [0, -1]])
    _Y = _X*_Z
    _S = Qu((2, 2), 'ud', [[1, 0], [0, cyclotomic(4)]])
    _T = Qu((2, 2), 'ud', [[1, 0], [0, cyclotomic(8)]])
    _Ti = _T.dag()
    assert _T * _Ti == _I
    _H = (1./sqrt(2))*Qu((2, 2), 'ud', [[1, 1], [1, -1]])

    pauli = build_algebra("IXZY",
        "X*X=I Z*Z=I Y*Y=-I X*Z=Y Z*X=-Y X*Y=Z Y*X=-Z Z*Y=-X Y*Z=X")

    I = pauli.I
    X = pauli.X
    Y = pauli.Y
    Z = pauli.Z

    S, Si, T, Ti, H = list(promote_pauli(pauli, [_S, _S.dag(), _T, _Ti, _H]))
    assert S*Si == I
    assert Si*S == I
    assert T*Ti == I
    assert Ti*T == I
    K = S*H # the "Bravyi-Kitaev T" gate
    Ki = H*Si
    assert K*Ki == I

    if argv.gauge:
        P = get_gauge(pauli)
    else:
        code = build_code(pauli)
        P = code.get_projector()
        m = len(code.ops) # number of independent stab gen
        print("m =", m)
        r = 2**(-m)
        P = r*P
        assert P*P == P

    n = P.grade

    #Xn = reduce(matmul, [X]*n)
    #print("K transverse:", Kn*P == P*Kn)
    #print(P*Kn == Kn*P)

    gate = argv.get("gate", "S")

    if gate == "X":
        Xn = reduce(matmul, [X]*n)
        print("go:")
        Q = Xn*P*Xn
    elif gate == "S": # clifford
        Sn = reduce(matmul, [S]*n)
        Sin = reduce(matmul, [Si]*n)
        print("go:")
        Q = Sn*P*Sin
    elif gate == "K": # clifford
        Kn = reduce(matmul, [K]*n)
        Kin = reduce(matmul, [Ki]*n)
        print("go:")
        Q = Kn*P*Kin
    elif gate == "T": # non-clifford
        Tn = reduce(matmul, [T]*n)
        Tin = reduce(matmul, [Ti]*n)
        print("go:")
        Q = Tn*P*Tin
    else:
        assert 0, gate

    print("Q:")
    print(Q)
    print("Q==P:", Q == P)

    QQ = (Q*Q)
    #print("Q*Q:")
    #print(QQ)

    assert QQ == Q



def main_1():

    I = Qu((2, 2), 'ud', [[1, 0], [0, 1]])
    X = Qu((2, 2), 'ud', [[0, 1], [1, 0]])
    Z = Qu((2, 2), 'ud', [[1, 0], [0, -1]])
    Y = X*Z
    T = Qu((2, 2), 'ud', [[1, 0], [0, cyclotomic(8)]])


    op = get_projector()
    n = op.grade

    if argv.show:
        print(op)

    P = op.evaluate({"I":I, "X":X, "Z":Z, "Y":Y})

    Tn = reduce(matmul, [T]*n)

    TnP = Tn*P
    PTn = P*Tn
    print(numpy.abs(TnP.v - PTn.v).sum())

    if Tn*P == P*Tn:
        print("transversal T")
    else:
        print("no transversal T")


if __name__ == "__main__":

    main()


