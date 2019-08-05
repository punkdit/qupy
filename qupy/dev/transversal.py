#!/usr/bin/env python3

from functools import reduce
from operator import mul, matmul, add
from math import sqrt

import numpy

from qupy.dense import Qu
from qupy.dev.su2 import (StabilizerCode, build_code, EPSILON, cyclotomic,
    promote_pauli, decompose)

from qupy.argv import argv

if argv.fast:
    #print("importing _algebra")
    from qupy.dev._algebra import Algebra, build_algebra, Tensor
else:
    from qupy.dev.algebra import Algebra, build_algebra, Tensor



def get_projector():
    pauli = build_algebra("IXZY",
        "X*X=I Z*Z=I Y*Y=-I X*Z=Y Z*X=-Y X*Y=Z Y*X=-Z Z*Y=-X Y*Z=X")

    I = pauli.I
    X = pauli.X
    Y = pauli.Y
    Z = pauli.Z

    op = build_code(pauli)
    return op


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

    m = argv.get("m", 4) # number of independent stab gen
    r = 2**(-m)

    P = build_code(pauli)
    n = P.grade
    #print(P)
    #print(P*P)

    P = r*P
    assert P*P == P

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


