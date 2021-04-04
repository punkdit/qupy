#!/usr/bin/env python3

from qupy.dev._algebra import Algebra, build_algebra, Tensor


EPSILON=1e-8

def test_real_pauli():
    #algebra = Algebra("IXYZ")
    pauli = build_algebra("IXZY",
        "X*X=I Z*Z=I Y*Y=-I X*Z=Y Z*X=-Y X*Y=Z Y*X=-Z Z*Y=-X Y*Z=X")

    I = pauli.I
    X = pauli.X
    Y = pauli.Y
    Z = pauli.Z

    op = Tensor(pauli)
    assert op.get_keys() == []
    op[(0,)] = 1.
    assert op[(0,)] == 1.
    assert op[(1,)] == 0.
    assert op.get_keys() == [(0,)]

    assert str(op) == "I", repr(str(op))
    op = I*X
    assert str(op) == "X"

    zero = Tensor(pauli)
    assert zero.norm() == 0.
    assert (X-X) == zero
    assert str(X-X) == "0", str(X-X)
    assert (X-X).norm() < EPSILON, (X-X).norm()
    assert X==X

    assert I*X == X
    assert X*X==I
    assert Z*Z==I
    assert Y*Y==-I
    assert X*Z==Y
    assert Z*X==-Y
    assert X*Y==Z
    assert Y*X==-Z
    assert Z*Y==-X
    assert Y*Z==X

    II = I@I
    XI = X@I
    IX = I@X
    XX = X@X
    assert II+II == 2*II

    assert X@(XI + IX) == X@X@I + X@I@X

    assert ((I-Y)@I + I@(I-Y)) == 2*I@I - I@Y - Y@I

    assert (XI + IX).subs({"X": I-Y}) == ((I-Y)@I + I@(I-Y))

    assert (X@I@Z).permute((1, 0, 2)) == I@X@Z

    assert XI.nnz(EPSILON) == 1
    assert (XI + IX).nnz(EPSILON) == 2


def test_complex_pauli():
    #algebra = Algebra("IXYZ")
    pauli = build_algebra("IXZY",
        "I*I=I I*X=X I*Z=Z I*Y=Y X*I=X X*X=I X*Z=-1j*Y"
        " X*Y=1j*Z Z*I=Z Z*X=1j*Y Z*Z=I Z*Y=-1j*X Y*I=Y Y*X=-1j*Z Y*Z=1j*X Y*Y=I")

    I = pauli.I
    X = pauli.X
    Y = pauli.Y
    Z = pauli.Z

    op = Tensor(pauli)
    assert op.get_keys() == []
    op[(0,)] = 1.
    assert op[(0,)] == 1.
    assert op[(1,)] == 0.
    assert op.get_keys() == [(0,)]

    assert str(op) == "I", repr(str(op))
    op = I*X
    assert str(op) == "X"

    zero = Tensor(pauli)
    assert zero.norm() == 0.
    assert (X-X) == zero
    assert str(X-X) == "0", str(X-X)
    assert (X-X).norm() < EPSILON, (X-X).norm()
    assert X==X

    assert I*X==X
    assert X*X==I
    assert Z*Z==I
    assert Y*Y==I
    assert X*Z==-1j*Y
    assert Z*X==1j*Y
    assert X*Y==1j*Z
    assert Y*X==-1j*Z
    assert Z*Y==-1j*X
    assert Y*Z==1j*X



if __name__ == "__main__":

    test_real_pauli()
    test_complex_pauli()

    print("OK")


