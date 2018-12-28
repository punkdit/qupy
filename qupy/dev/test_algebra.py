#!/usr/bin/env python3

from qupy.dev._algebra import Algebra, build_algebra, Tensor


EPSILON=1e-8

def test_pauli():
    #algebra = Algebra("IXYZ")
    pauli = build_algebra("IXZY",
        "X*X=I Z*Z=I Y*Y=-I X*Z=Y Z*X=-Y X*Y=Z Y*X=-Z Z*Y=-X Y*Z=X")

    I = pauli.I
    X = pauli.X
    Y = pauli.Y
    Z = pauli.Z

    op = Tensor(pauli)
    op[(0,)] = 1.
    assert op[(0,)] == 1.
    assert op[(1,)] == 0.

    assert str(op) == "I"
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


if __name__ == "__main__":

    test_pauli()


