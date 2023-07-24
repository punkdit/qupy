#!/usr/bin/env python3

from qupy.dev._algebra import Algebra, build_algebra, Tensor


EPSILON=1e-8

def test_complex_pauli():
    #algebra = Algebra("IXYZ")
    pauli = build_algebra("IXZY",
        "I*I=I I*X=X I*Z=Z I*Y=Y X*I=X X*X=I X*Z=-1i*Y"
        " X*Y=1i*Z Z*I=Z Z*X=1i*Y Z*Z=I Z*Y=-1i*X Y*I=Y Y*X=-1i*Z Y*Z=1i*X Y*Y=I")

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

    n = 5
    stabs = [
        X@Z@Z@X@I,
        I@X@Z@Z@X,
        X@I@X@Z@Z,
        Z@X@I@X@Z,
        Z@Z@X@I@X,
    ]
    logops = [
        X@X@X@X@X,
        Z@Z@Z@Z@Z,
    ]
    
    for s in stabs:
     for t in stabs:
        assert s*t == t*s

    basis = [I, X, Z, Y]
    for a in basis:
     for b in basis:
      for c in basis:
       for d in basis:
        for e in basis:
            op = a@b@c@d@e
            sig = [int(h*op!=op*h) for h in stabs+logops]
            if sig == [0,0,0,1,1,0,0]:
                sop = str(op)
                if sop.count("Y")==2 and sop.count("I")==3:
                    print(sop)
            #else:
            #    print(sig)

    from qupy.ldpc.css import CSSCode
    from qupy.ldpc.solve import parse, dot2
    Hx = parse("""
    X..X..XX..
    .X..X..XX.
    X.X.....XX
    .X.X.X...X
    """)
    Tz = parse("""
    ...ZZ...ZZ
    Z..Z.Z..Z.
    .Z.Z..Z.Z.
    ..ZZ...ZZ.
    """)
    Hz = parse("""
    .ZZ..Z..Z.
    ..ZZ..Z..Z
    ...ZZZ.Z..
    Z...Z.Z.Z.
    """)
    Tx = parse("""
    ...XX...XX
    X..X.X..X.
    .X.X..X.X.
    ..XX...XX.
    """)
    Lx = parse("""
    XXXXX.....
    .....XXXXX
    """)
    Lz = parse("""
    ZZZZZ.....
    .....ZZZZZ
    """)

    code = CSSCode(Hx=Hx, Hz=Hz, Tx=Tx, Tz=Tz, Lx=Lx, Lz=Lz, check=True)
    print(code.longstr())




if __name__ == "__main__":

    test_complex_pauli()

    print("OK")


