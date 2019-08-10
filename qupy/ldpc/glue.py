#!/usr/bin/env python3

import numpy
import random

from qupy.ldpc.solve import array2, parse, dot2, compose2, eq2, zeros2
from qupy.ldpc.solve import rand2, find_kernel, image, identity2, rank, shortstr
from qupy.ldpc.solve import shortstrx
from qupy.ldpc import solve
from qupy.ldpc.css import CSSCode
from qupy.ldpc.chain import Chain, Morphism
from qupy.ldpc import chain
from qupy.ldpc.gallagher import classical_distance

from qupy.argv import argv


def fstr(H):
    s = str(H).replace("0", ".")
    lines = s.split('\n')
    lines = [("%2d "%i)+line for (i, line) in enumerate(lines)]
    return '\n'.join(lines)


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
    CA = Morphism(C, A, [CAx, CAn, CAz])

    # Chain map from C -> B
    CBz = CAz
    CBn = CAn
    CBx = CAx
    CB = Morphism(C, B, [CBx, CBn, CBz])

    AD, BD, D, _ = chain.pushout(CA, CB)
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
    CD = Morphism(C, D, [CDx, CDn, CDz])

    _, _, E, _ = chain.pushout(CA, CD)
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
    CA = Morphism(C, A, [CAx, CAn, CAz])

    # Chain map from C -> B
    CBz = CAz
    CBn = CAn
    CBx = CAx
    CB = Morphism(C, B, [CBx, CBn, CBz])

    AD, BD, D, _ = chain.pushout(CA, CB)
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
    CA = Morphism(C, A, [CAm, CAn])

    CBm = zeros2(0, 1)
    CBn = zeros2(0, 1)
    CB = Morphism(C, B, [CBm, CBn])

    AD, BD, D, _ = chain.pushout(CA, CB)
    assert eq2(D[0], array2([[1,1,1], [0,1,1]])) # glue two checks at a bit

    # --------------------------

    A = Chain([H.transpose()])
    B = Chain([zeros2(0, 0)])
    C = Chain([zeros2(1, 0)])

    CAn = zeros2(n, 1)
    CAn[0, 0] = 1
    CAn[n-1, 0] = 1
    CAm = zeros2(m, 0)
    CA = Morphism(C, A, [CAn, CAm])

    CBn = zeros2(0, 1)
    CBm = zeros2(0, 0)
    CB = Morphism(C, B, [CBn, CBm])

    AD, BD, D, _ = chain.pushout(CA, CB)
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
    f = Morphism(C, A, [fm, fn])

    gm = zeros2(m, 1)
    gm[0, 0] = 1
    gn = zeros2(n, 1)
    gn[0, 0] = 1
    g = Morphism(C, A, [gm, gn])

    AD, BD, D = chain.equalizer(f, g)
    assert eq2(D[0], array2([[1,1,1], [0,1,1]])) # glue two checks at a bit

    # --------------------------

    A = Chain([H.transpose()])
    C = Chain([zeros2(1, 0)])

    fn = zeros2(n, 1)
    fn[0, 0] = 1
    fm = zeros2(m, 0)
    f = Morphism(C, A, [fn, fm])

    gn = zeros2(n, 1)
    gn[n-1, 0] = 1
    gm = zeros2(m, 0)
    g = Morphism(C, A, [gn, gm])

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
    C1 = Morphism(C, A1, [C1m, C1n])

    C2n = zeros2(n2, 1)
    C2n[i2, 0] = 1
    C2m = dot2(H2, C2n)
    C2 = Morphism(C, A2, [C2m, C2n])

    AD, BD, D, _ = chain.pushout(C1, C2)

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
    f = Morphism(C, A, [fm, fn])

    gn = zeros2(n, 1)
    gn[i2, 0] = 1
    gm = dot2(H, gn)
    g = Morphism(C, A, [gm, gn])

    _, _, D = chain.equalizer(f, g)

    H = D[0]
    #print(H.shape)
    #print(H)
    return H


def test_glue():

    m = argv.get("m", 9)
    n = argv.get("n", 10)

    d = argv.get("d", 0)
    p = argv.get("p", 0.5)
    weight = argv.weight

    H1 = rand2(m, n, p, weight)
    G1 = find_kernel(H1)
    G1t = G1.transpose()
    H1t = H1.transpose()
    A1 = Chain([G1, H1t])
    k1 = len(G1)

    print("H1")
    print(fstr(H1))
    print()
    print(fstr(G1))

    w = wenum(H1)
    print("wenum:", [len(wi) for wi in w])

    H2 = rand2(m, n, p, weight)
    H2t = H2.transpose()
    G2 = find_kernel(H2)
    G2t = G2.transpose()
    A2 = Chain([G2, H2t])
    k2 = len(G2)

    print("H2")
    print(fstr(H2))
    print()
    print(fstr(G2))

    w = wenum(H2)
    print("wenum:", [len(wi) for wi in w])

    if k1 != k2:
        return
    k = k1

    I = identity2(k)
    B = Chain([I, zeros2(k, 0)])

    a = zeros2(n, k)
    for i in range(k):
        a[i, i] = 1
    f1 = Morphism(B, A1, [dot2(G1, a), a, zeros2(m, 0)])
    f2 = Morphism(B, A2, [dot2(G2, a), a, zeros2(m, 0)])

    a, b, C, _ = chain.pushout(f1, f2)

    H = C[1].transpose()
    print("H:")
    print(fstr(H))

    w = wenum(H)
    print("wenum:", [len(wi) for wi in w])




def test_color():

    HxA = array2([[1,1,0,0],[1,0,1,0],[1,1,1,1]])
    HzA = array2([[1,1,1,1]])
    A = Chain([HxA, HzA.transpose()])

    HxB = HxA
    HzB = HzA
    B = Chain([HxB, HzB.transpose()])

    HzC = zeros2(0, 2)
    HxC = array2([[1,0],[0,1]])
    C = Chain([HxC, HzC.transpose()])

    # Chain map from C -> A
    CAz = zeros2(1, 0)
    CAn = zeros2(4, 2)
    CAn[0, 0] = 1
    CAn[1, 1] = 1
    CAx = dot2(HxA, CAn)
    #print(CAx)
    CA = Morphism(C, A, [CAx, CAn, CAz])

    # Chain map from C -> B
    CBz = CAz
    CBn = CAn
    CBx = CAx
    CB = Morphism(C, B, [CBx, CBn, CBz])

    AD, BD, D, _ = chain.pushout(CA, CB)
    code = D.get_code()
    print(code)
    print("A --> D")
    print(fstr(AD[0]))
    print("-----------")
    print(fstr(AD[1]))
    print("B --> D")
    print(fstr(BD[0]))
    print("-----------")
    print(fstr(BD[1]))
    print("Hx:")
    print(fstr(code.Hx))

    # argh...


def rand_full_rank(m, n=None):
    if n is None:
        n=m
    assert n>=m
    while 1:
        f = rand2(m, n)
        if rank(f) == m:
            break
    return f


#def rand_iso(achain):
    



def test_universal():

    for trial in range(100):

        m, n = 3, 4
        J = rand2(m, n)
        K = rand2(m, n)
    
        a = Chain([J])
        b = Chain([K])
        amorph = a.from_zero()
        bmorph = b.from_zero()
    
        am, bm, c, u = chain.pushout(amorph, bmorph)
        assert u is None
    
        C = c[0]
        mm, nn = C.shape
    
        f = rand_full_rank(nn-2, nn)
        g, H, _ = solve.pushout(C, f)
    
        _c = Chain([H])
        m = Morphism(c, _c, [g, f])
        assert m*am is not None
        assert m*bm is not None
    
        _, _, _, u = chain.pushout(amorph, bmorph, m*am, m*bm, _c)
        assert u is not None
        assert u==m



def make_ldpc(m, n, p=0.5, weight=None, dist=0):
    while 1:
        H = rand2(m, n, p, weight)
        d = classical_distance(H, dist)
        if d>=dist:
            break
    return H


def glue_pairs(H1, H2, pairs):

    m1, n1 = H1.shape
    m2, n2 = H2.shape
    k = len(pairs)

    A1 = Chain([H1])
    A2 = Chain([H2])
    C  = Chain([identity2(k)])

    C1n = zeros2(n1, k)
    for idx, pair in enumerate(pairs):
        i, j = pair
        C1n[i, idx] = 1
    C1m = dot2(H1, C1n)
    C1 = Morphism(C, A1, [C1m, C1n])

    C2n = zeros2(n2, k)
    for idx, pair in enumerate(pairs):
        i, j = pair
        C2n[j, idx] = 1
    C2m = dot2(H2, C2n)
    C2 = Morphism(C, A2, [C2m, C2n])

    AD, BD, D, _ = chain.pushout(C1, C2)

    H = D[0]
    #print(H.shape)
    #print(H)
    return H


def ldpc_str(H):
    m, n = H.shape
    assert rank(H) == m
    k = n-m
    d = classical_distance(H)
    return "[%d, %d, %d]" % (n, k, d)


def test_ldpc():

    n = argv.get("n", 14)
    m = argv.get("m", n-3)

    d = argv.get("d", 0)
    p = argv.get("p", 0.5)
    weight = argv.get("weight", 4)
    dist = argv.get("dist", 4)

    H1 = make_ldpc(m, n, p, weight, dist)
    #print(fstr(H1))

    #Gt = find_kernel(H1)
    #w = wenum(H1)
    #print("wenum:", [len(wi) for wi in w])

    H2 = make_ldpc(m, n, p, weight, dist)
    #print(fstr(H2))

    k = argv.get("k", 3)
    pairs = [(i, i) for i in range(k)]
    K = glue_pairs(H1, H2, pairs)
    #print(fstr(K))
    assert rank(K) == len(K)

    print(ldpc_str(H1), "+", ldpc_str(H2), "=", ldpc_str(K))

    if argv.show:
        print(shortstr(K))



def glue_quantum(Hx1, Hz1, Hx2, Hz2, pairs):

    mx1, n1 = Hx1.shape
    mx2, n2 = Hx2.shape
    mz1, _ = Hz1.shape
    mz2, _ = Hz2.shape
    k = len(pairs)

    A1 = Chain([Hz1, Hx1.transpose()])
    A2 = Chain([Hz2, Hx2.transpose()])
    C  = Chain([identity2(k), zeros2(k, 0)])

    C1n = zeros2(n1, k)
    for idx, pair in enumerate(pairs):
        i, j = pair
        C1n[i, idx] = 1
    C1m = dot2(Hz1, C1n)
    C1 = Morphism(C, A1, [C1m, C1n, zeros2(mx1, 0)])

    C2n = zeros2(n2, k)
    for idx, pair in enumerate(pairs):
        i, j = pair
        C2n[j, idx] = 1
    C2m = dot2(Hz2, C2n)
    C2 = Morphism(C, A2, [C2m, C2n, zeros2(mx2, 0)])

    AD, BD, D, _ = chain.pushout(C1, C2)

    Hz, Hxt = D[0], D[1]
    #print(H.shape)
    #print(H)

    return Hz, Hxt.transpose()


def glue1_quantum(Hx, Hz, i1, i2):
    assert i1!=i2

    mx, n = Hx.shape
    mz, _ = Hz.shape
    k = 1

    A = Chain([Hz, Hx.transpose()])
    C  = Chain([identity2(k), zeros2(k, 0)])

    fn = zeros2(n, 1)
    fn[i1, 0] = 1
    fm = dot2(Hz, fn)
    f = Morphism(C, A, [fm, fn, zeros2(mx, 0)])

    gn = zeros2(n, 1)
    gn[i2, 0] = 1
    gm = dot2(Hz, gn)
    g = Morphism(C, A, [gm, gn, zeros2(mx, 0)])

    _, _, D = chain.equalizer(f, g)

    Hz, Hxt = D[0], D[1]
    return Hz, Hxt.transpose()


def make_q(n, m):
    k = n-2*m
    assert k>=0 

    assert m>0
    Hx = [rand2(1, n)[0]]
    Hz = []
    while 1:
        _Hx = array2(Hx)
        while 1:
            v = rand2(1, n)
            if dot2(Hx, v.transpose()).sum() == 0:
                break
        Hz.append(v[0])
        if len(Hz)==m:
            break
        while 1:
            v = rand2(1, n)
            if dot2(Hz, v.transpose()).sum() == 0:
                break
        Hx.append(v[0])
    Hx = array2(Hx)
    Hz = array2(Hz)
    assert dot2(Hx, Hz.transpose()).sum() == 0

    return Hx, Hz


def make_quantum(n, m, dist=0):
    while 1:
        Hx, Hz = make_q(n, m)
        if rank(Hx) < m or rank(Hz) < m:
            continue
        d = classical_distance(Hx, dist)
        if d < dist:
            continue
        d = classical_distance(Hz, dist)
        if d < dist:
            continue
        break
    return Hx, Hz


def test_quantum():

    m = argv.get("m", 4)
    n = argv.get("n", m+m+1)
    dist = argv.get("dist", 0)
    N = argv.get("N", 2)

    codes = []
    code = None
    for i in range(N):
        Hx, Hz = make_quantum(n, m, dist)
        print("Hx, Hz:")
        print(shortstrx(Hx, Hz))
        c = CSSCode(Hx=Hx, Hz=Hz)
        codes.append(c)
        code = c if code is None else code + c

    print(code)
    code.save("glue.ldpc")

    return

    code = CSSCode(Hx=Hx, Hz=Hz)
    print(code)
    code = code+code
    print(shortstrx(code.Hx, code.Hz))
    #Hx, Hz = glue1_quantum(code.Hx, code.Hz, 0, 1)
    #code = CSSCode(Hx=Hx, Hz=Hz)
    code = code.glue(0, n)
    print(code)
    print(shortstrx(code.Hx, code.Hz))
    return

    k = argv.get("k", 1)
    pairs = [(i, i) for i in range(k)]
    
    H1x, H1z = glue_quantum(Hx, Hz, Hx, Hz, pairs)
    assert dot2(H1x, H1z.transpose()).sum() == 0

    code = CSSCode(Hx=H1x, Hz=H1z)
    print(code)
    print(shortstrx(code.Hx, code.Hz))


if __name__ == "__main__":

    _seed = argv.get("seed")
    if _seed is not None:
        numpy.random.seed(_seed)
        random.seed(_seed)

#    test()
#    test_selfdual()
#    test_colimit()
#    test_equalizer()
#    test_glue()
#    test_color()
#    test_universal()
#    test_ldpc()
    test_quantum()

    print("OK")


