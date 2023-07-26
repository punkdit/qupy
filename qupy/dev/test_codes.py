#!/usr/bin/env python3

from time import time
start_time = time()

from functools import reduce
from operator import matmul, mul, add

from qupy.dev._algebra import Algebra, build_algebra, Tensor
from qupy.ldpc.css import CSSCode
from qupy.ldpc.solve import parse, dot2
from qupy.argv import argv

EPSILON=1e-8
pauli = build_algebra("IXZY",
    "I*I=I I*X=X I*Z=Z I*Y=Y X*I=X X*X=I X*Z=-1i*Y"
    " X*Y=1i*Z Z*I=Z Z*X=1i*Y Z*Z=I Z*Y=-1i*X Y*I=Y Y*X=-1i*Z Y*Z=1i*X Y*Y=I")

I = pauli.I
X = pauli.X
Y = pauli.Y
Z = pauli.Z

T = (0.8535533905932737+0.35355339059327373j)*I+\
    (0.1464466094067262-0.35355339059327373j)*Z
Ti = (0.8535533905932737-0.35355339059327373j)*I+\
    (0.1464466094067262+0.35355339059327373j)*Z


def test_complex_pauli():
    #algebra = Algebra("IXYZ")
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


#class QCode(object):
#    def __init__(self, H, L):

make_zop = lambda h: reduce(matmul, [[I, Z][hi] for hi in h])
make_xop = lambda h: reduce(matmul, [[I, X][hi] for hi in h])


def test_832():
    """
    _Transversal T gate on the [[8,3,2]] colour code.
    https://earltcampbell.com/2016/09/26/the-smallest-interesting-colour-code
    https://arxiv.org/abs/1706.02717
    """

    Hz = parse("1111.... 11..11.. 1.1.1.1. 11111111")
    Hx = parse("11111111")
    Lz = parse("1...1... 1.1..... 11......")
    Lx = parse("1111.... 11..11.. 1.1.1.1.")

    mx, n = Hx.shape
    stabs = [make_zop(h) for h in Hz] + [make_xop(h) for h in Hx]

    for a in stabs:
      for b in stabs:
        assert a*b == b*a

    In = reduce(matmul, [I]*n)
    P = reduce(mul, [In+op for op in stabs])

    assert T*Ti == I
    assert (T*T*T*T) == Z

    Tn  = reduce(matmul, [[T,Ti][0] for i in range(n)])
    Tni = reduce(matmul, [[Ti,T][0] for i in range(n)])
    assert Tn*Tni == In
    Xn = reduce(matmul, [X]*n)

    op = Tni*P*Tn
    assert op == P

    # --------------------------------------------------------------

    n = 4
    Hz = parse("1111 1.1. 11..")
    Hx = parse("1111")

    mx, n = Hx.shape
    stabs = [make_zop(h) for h in Hz] + [make_xop(h) for h in Hx]
    
    for a in stabs:
      for b in stabs:
        assert a*b == b*a

    In = reduce(matmul, [I]*n)
    P = reduce(mul, [In+op for op in stabs])
    assert str(P) == """
    IIII+IIZZ+IZIZ+IZZI+XXXX-XXYY-XYXY-XYYX+ZIIZ+ZIZI+ZZII+ZZZZ-YXXY-YXYX-YYXX+YYYY
    """.strip()

    Tn  = reduce(matmul, [[T,Ti][i%2] for i in range(n)])
    Tni = reduce(matmul, [[Ti,T][i%2] for i in range(n)])
    assert Tn*Tni == In
    Xn = reduce(matmul, [X]*n)

    op = Tni*P*Tn
    assert op==P

    # --------------------------------------------------------------

    n = 2
    Hz = parse("11")
    Hx = parse("11")

    mx, n = Hx.shape
    stabs = [make_zop(h) for h in Hz] + [make_xop(h) for h in Hx]
    
    for a in stabs:
      for b in stabs:
        assert a*b == b*a

    In = reduce(matmul, [I]*n)
    P = reduce(mul, [In+op for op in stabs])
    assert str(P) == "II+XX+ZZ-YY"

    Tn  = reduce(matmul, [[T,Ti][i%2] for i in range(n)])
    Tni = reduce(matmul, [[Ti,T][i%2] for i in range(n)])
    assert Tn*Tni == In
    Xn = reduce(matmul, [X]*n)

    op = Tni*P*Tn
    assert op==P





def test_unwrap():


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
                    assert sop == 'IIYYI'
            #else:
            #    print(sig)


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

    _seed = argv.get("seed")
    if _seed is not None:
        print("seed(%s)"%_seed)
        seed(_seed)

    profile = argv.profile
    fn = argv.next() or "test_complex_pauli"

    print("%s()"%fn)

    if profile:
        import cProfile as profile
        profile.run("%s()"%fn)

    else:
        fn = eval(fn)
        fn()

    print("\nOK: finished in %.3f seconds"%(time() - start_time))
    print()





