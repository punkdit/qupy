#!/usr/bin/env python3

import os

import numpy
from numpy import random as ra
from numpy import concatenate as cat
import random

from qupy.ldpc.solve import array2, parse, dot2, compose2, eq2, zeros2
from qupy.ldpc.solve import rand2, find_kernel, image, identity2, rank, shortstr
from qupy.ldpc.solve import shortstrx, pseudo_inverse, linear_independent, span
from qupy.ldpc import solve
from qupy.ldpc.css import CSSCode
from qupy.ldpc.chain import Chain, Morphism
from qupy.ldpc import chain
from qupy.ldpc.gallagher import classical_distance
from qupy.ldpc.bpdecode import RadfordBPDecoder
from qupy.ldpc import reed_muller

from qupy.argv import argv
from qupy.tool import cross, choose


def is_morthogonal(G, m):
    k = len(G)
    if m==1:
        for v in G:
            if v.sum()%2 != 0:
                return False
        return True
    if m>2 and not is_morthogonal(G, m-1):
        return False
    items = list(range(k))
    for idxs in choose(items, m):
        v = G[idxs[0]]
        for idx in idxs[1:]:
            v = v * G[idx]
        if v.sum()%2 != 0:
            return False
    return True


def strong_morthogonal(G, m):
    k = len(G)
    assert m>=1
    if m==1:
        for v in G:
            if v.sum()%2 != 0:
                return False
        return True
    if not strong_morthogonal(G, m-1):
        return False
    items = list(range(k))
    for idxs in choose(items, m):
        v = G[idxs[0]]
        for idx in idxs[1:]:
            v = v * G[idx]
        if v.sum()%2 != 0:
            return False
    return True




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
    return Hxt.transpose(), Hz


def glue_self(Hx, Hz, pairs):
    mx, n = Hx.shape
    mz, _ = Hz.shape
    k = len(pairs)

    A = Chain([Hz, Hx.transpose()])
    C  = Chain([identity2(k), zeros2(k, 0)])

    fn = zeros2(n, k)
    for idx, pair in enumerate(pairs):
        i1, i2 = pair
        fn[i1, idx] = 1
    fm = dot2(Hz, fn)
    f = Morphism(C, A, [fm, fn, zeros2(mx, 0)])

    gn = zeros2(n, k)
    for idx, pair in enumerate(pairs):
        i1, i2 = pair
        gn[i2, idx] = 1
    gm = dot2(Hz, gn)
    g = Morphism(C, A, [gm, gn, zeros2(mx, 0)])

    _, _, D = chain.equalizer(f, g)

    Hz, Hxt = D[0], D[1]
    return Hxt.transpose(), Hz


def make_q(n, m, weight=None):
    k = n-2*m
    assert k>=0 

    assert m>0
    Hx = [rand2(1, n, weight=weight)[0]]
    Hz = []
    while 1:
        _Hx = array2(Hx)
        while 1:
            v = rand2(1, n, weight=weight)
            if dot2(Hx, v.transpose()).sum() == 0:
                break
        Hz.append(v[0])
        if len(Hz)==m:
            break
        while 1:
            v = rand2(1, n, weight=weight)
            if dot2(Hz, v.transpose()).sum() == 0:
                break
        Hx.append(v[0])
    Hx = array2(Hx)
    Hz = array2(Hz)
    assert dot2(Hx, Hz.transpose()).sum() == 0

    return Hx, Hz


def make_quantum(n, m, dist=0, weight=None):
    while 1:
        Hx, Hz = make_q(n, m, weight)
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


def score(code, p=0.05, N=10000):
    # each bit gets a score: higher is worse
    decoder = RadfordBPDecoder(2, code.Hz)
    n = code.n
    counts = numpy.zeros(n, dtype=int)

    err = 0
    for trial in range(N):
        err_op = ra.binomial(1, p, (code.n,))
        err_op = err_op.astype(numpy.int32)
        #write(str(err_op.sum()))
        s = dot2(code.Hz, err_op) # syndrome
        #write(":s%d:"%s.sum())
        op = decoder.decode(p, err_op, verbose=False)
        success = False
        if op is not None:
            op = (op+err_op)%2
            # Should be a codeword of Hz (kernel of Hz)
            assert dot2(code.Hz, op).sum() == 0
            #write("%d:"%op.sum())
            if dot2(code.Lz, op).sum():
#                counts += op
                counts += err_op
            else:
                success = True
        if not success:
            err += 1

    return counts, (err / N)

def make_surface():
    Hx = array2([[1,1,1,0,0],[0,0,1,1,1]])
    Hz = array2([[1,0,1,1,0],[0,1,1,0,1]])
    return Hx, Hz


def test_quantum():

    m = argv.get("m", 4)
    n = argv.get("n", m+m+1)
    dist = argv.get("dist", 3)
    N = argv.get("N", 2)
    M = argv.get("M", N)
    p = argv.get("p", 0.03)
    weight = argv.weight

    codes = []
    code = None
    for i in range(N):
        Hx, Hz = make_quantum(n, m, dist, weight)
        #Hx, Hz = make_surface()
        print("Hx, Hz:")
        print(shortstrx(Hx, Hz))
        c = CSSCode(Hx=Hx, Hz=Hz)
        codes.append(c)
        code = c if code is None else code + c

    for _ in range(2*M):
        print(code)
        #code.save("glue.ldpc")
        counts, err = score(code, p)
        print("err = %.4f"%err)
        i0 = numpy.argmin(counts)
        i1 = numpy.argmax(counts)
        assert i0 != i1
        print(counts, i0, i1)
        code = code.glue(i0, i1)
        code = code.dual()

    print(shortstrx(code.Hx, code.Hz))
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


def glue_logops():

    m = argv.get("m", 4)
    n = argv.get("n", m+m+1)
    dist = argv.get("dist", 3)
    N = argv.get("N", 2)
    M = argv.get("M", N)
    p = argv.get("p", 0.03)
    weight = argv.weight

    codes = []
    code = None
    for i in range(N):
        Hx, Hz = make_quantum(n, m, dist, weight)
        #Hx, Hz = make_surface()
        print("Hx, Hz:")
        print(shortstrx(Hx, Hz))
        c = CSSCode(Hx=Hx, Hz=Hz)
        codes.append(c)
        code = c if code is None else code + c

    A, B = codes
    code = A+B
    
    print(code)
    print("Lx, Lz:")
    print(shortstrx(code.Lx, code.Lz))
    print("Hx, Hz:")
    print(shortstrx(code.Hx, code.Hz))
    print()

    #Hx = cat((code.Lx, code.Hx))
    Hx = code.Hx
    Hz = cat((code.Lz, code.Hz))

    idxs = list(range(2*n))
    idxs.sort(key = lambda i : (-code.Lz[:, i].sum(),))
    Hx = Hx[:, idxs]
    Hz = Hz[:, idxs]

    print(shortstrx(Hx, Hz))

    i0 = argv.get("i0")
    i1 = argv.get("i1")
    if i0 is None:
        return

#    code = code.glue(0, n)
#    print(code)
#    print(shortstrx(code.Hx, code.Hz))

    Hx, Hz = glue1_quantum(Hx, Hz, i0, i1)
    print("Hx, Hz:")
    print(shortstrx(Hx, Hz))


def make_morthogonal(m, n, genus):
    while 1:
        G = rand2(m, n)
        if strong_morthogonal(G, genus) and rank(G)==m and numpy.min(G.sum(0)):
            break
    return G


def direct_sum(A, B):
    C = zeros2(A.shape[0] + B.shape[0], A.shape[1] + B.shape[1])
    C[:A.shape[0], :A.shape[1]] = A
    C[A.shape[0]:, A.shape[1]:] = B
    return C


def find_triorth(m, k):
    # Bravyi, Haah, 1209.2426v1 sec IX.
    # https://arxiv.org/pdf/1209.2426.pdf

    verbose = argv.get("verbose")
    #m = argv.get("m", 6) # _number of rows
    #k = argv.get("k", None) # _number of odd-weight rows

    # these are the variables N_x
    xs = list(cross([(0, 1)]*m))

    maxweight = argv.maxweight
    minweight = argv.get("minweight", 1)

    xs = [x for x in xs if minweight <= sum(x)]
    if maxweight:
        xs = [x for x in xs if sum(x) <= maxweight]

    N = len(xs)

    lhs = []
    rhs = []

    # bi-orthogonality
    for a in range(m):
      for b in range(a+1, m):
        v = zeros2(N)
        for i, x in enumerate(xs):
            if x[a] == x[b] == 1:
                v[i] = 1
        if v.sum():
            lhs.append(v)
            rhs.append(0)

    # tri-orthogonality
    for a in range(m):
      for b in range(a+1, m):
       for c in range(b+1, m):
        v = zeros2(N)
        for i, x in enumerate(xs):
            if x[a] == x[b] == x[c] == 1:
                v[i] = 1
        if v.sum():
            lhs.append(v)
            rhs.append(0)

#    # dissallow columns with weight <= 1
#    for i, x in enumerate(xs):
#        if sum(x)<=1:
#            v = zeros2(N)
#            v[i] = 1
#            lhs.append(v)
#            rhs.append(0)

    if k is not None:
      # constrain to k _number of odd-weight rows
      assert 0<=k<m
      for a in range(m):
        v = zeros2(N)
        for i, x in enumerate(xs):
          if x[a] == 1:
            v[i] = 1
        lhs.append(v)
        if a<k:
            rhs.append(1)
        else:
            rhs.append(0)

    A = array2(lhs)
    rhs = array2(rhs)
    #print(shortstr(A))

    B = pseudo_inverse(A)
    soln = dot2(B, rhs)
    if not eq2(dot2(A, soln), rhs):
        print("no solution")
        return
    if verbose:
        print("soln:")
        print(shortstr(soln))

    soln.shape = (N, 1)
    rhs.shape = A.shape[0], 1

    K = array2(list(find_kernel(A)))
    #print(K)
    #print( dot2(A, K.transpose()))
    #sols = []
    #for v in span(K):
    best = None
    density = 1.0
    size = 99*N
    trials = argv.get("trials", 1024)
    count = 0
    for trial in range(trials):
        u = rand2(len(K), 1)
        v = dot2(K.transpose(), u)
        #print(v)
        v = (v+soln)%2
        assert eq2(dot2(A, v), rhs)

        if v.sum() > size:
            continue
        size = v.sum()

        Gt = []
        for i, x in enumerate(xs):
            if v[i]:
                Gt.append(x)
        if not Gt:
            continue
        Gt = array2(Gt)
        G = Gt.transpose()
        assert is_morthogonal(G, 3)
        if G.shape[1]<m:
            continue

        if 0 in G.sum(1):
            continue

        if argv.strong_morthogonal and not strong_morthogonal(G, 3):
            continue

        #print(shortstr(G))
#        for g in G:
#            print(shortstr(g), g.sum())
#        print()

        _density = float(G.sum()) / (G.shape[0]*G.shape[1])
        #if best is None or _density < density:
        if best is None or G.shape[1] <= size:
            best = G
            size = G.shape[1]
            density = _density

        if 0:
            #sols.append(G)
            Gx = even_rows(G)
            assert is_morthogonal(Gx, 3)
            if len(Gx)==0:
                continue
            GGx = array2(list(span(Gx)))
            assert is_morthogonal(GGx, 3)

        count += 1

    print("found %d solutions" % count)
    if best is None:
        return

    G = best
    #print(shortstr(G))

    for g in G:
        print(shortstr(g), g.sum())
    print()
    print("density:", density)
    print("shape:", G.shape)

    G = linear_independent(G)

    if 0:
        A = list(span(G))
        print(strong_morthogonal(A, 1))
        print(strong_morthogonal(A, 2))
        print(strong_morthogonal(A, 3))

    G = [row for row in G if row.sum()%2 == 0]
    return array2(G)

    #print(shortstr(dot2(G, G.transpose())))

    if 0:
        B = pseudo_inverse(A)
        v = dot2(B, rhs)
        print("B:")
        print(shortstr(B))
        print("v:")
        print(shortstr(v))
        assert eq2(dot2(B, v), rhs)



def glue_self_classical(Hz, pairs):
    mz, n = Hz.shape
    k = len(pairs)

    A = Chain([Hz])
    C  = Chain([identity2(k)])

    fn = zeros2(n, k)
    for idx, pair in enumerate(pairs):
        i1, i2 = pair
        fn[i1, idx] = 1
    fm = dot2(Hz, fn)
    f = Morphism(C, A, [fm, fn])

    gn = zeros2(n, k)
    for idx, pair in enumerate(pairs):
        i1, i2 = pair
        gn[i2, idx] = 1
    gm = dot2(Hz, gn)
    g = Morphism(C, A, [gm, gn])

    _, _, D = chain.equalizer(f, g)

    Hz = D[0]
    return Hz



def glue_morth():

    m = argv.get("m", 4)
    n = argv.get("n", m+m+1)
    genus = argv.get("genus", 2)

    if 0:
        H = make_morthogonal(m, n, genus)
    elif 0:
        H = reed_muller.build(1, 4).G
        print(shortstrx(H))
    else:
        H = find_triorth(m, 1)

    assert dot2(H, H.transpose()).sum() == 0

    Hx = Hz = H
    if 0:
        print(classical_distance(Hx))
        print(classical_distance(Hz))
    i0 = 0
    i1 = Hx.shape[1]

    Hx = direct_sum(Hx, Hx)
    Hz = direct_sum(Hz, Hz)
    print(shortstrx(Hx, Hz))
    print(strong_morthogonal(Hx, genus))
    print(strong_morthogonal(Hz, genus))
    print()
    code = CSSCode(Hx=Hx, Hz=Hz)
    print(code)

    if 0:
        Hz = glue_self_classical(Hz, [(i0, i1), (i0, i1+1)])
        print(shortstrx(Hz))
        print(strong_morthogonal(Hz, genus))
        print()
        return

    Hx, Hz = glue_self(Hx, Hz, [(i0, i1), (i0, i1+1)])
    #Hx, Hz = glue_self(Hx, Hz, [(i0, i1)])
    print(shortstrx(Hx, Hz))
    print(strong_morthogonal(Hx, genus))
    print(strong_morthogonal(Hz, genus))
    print()
    code = CSSCode(Hx=Hx, Hz=Hz)

#    Hx, Hz = glue_self(Hx, Hz, [(0, n)])
#    print(shortstrx(Hx, Hz))
#    print(strong_morthogonal(Hx, genus))
#    print(strong_morthogonal(Hz, genus))
#    print()

    code = CSSCode(Hx=Hx, Hz=Hz)
    print(code)
    #print(classical_distance(Hx))
    #print(classical_distance(Hz))
    #print(code.longstr())


def glue_classical():

    from bruhat.triply_even import build

    genus = argv.get("genus", 3)

    m = argv.get("dim", 7)
    idx = argv.get("idx", 144)
    H = build.get(m, idx)
    H = H.astype(numpy.int32)
    n = H.shape[1]

    if argv.scramble:
        R = rand2(m, m)
        H = dot2(R, H)

    print(shortstrx(H))
    assert dot2(H, H.transpose()).sum() == 0

    i0 = argv.get("i0", 0)
    i1 = argv.get("i1", i0)
    i2 = argv.get("i2", n)
    i3 = argv.get("i3", i2+1)
    # glue i0<-->i2 and i1<-->i3

    #H2 = direct_sum(H, H)
    H2 = H
    #print(shortstrx(H2))
    assert strong_morthogonal(H2, genus)
    print()

    H3 = glue_self_classical(H2, [(i0, i2), (i1, i3)])
    print(shortstrx(H3))
    assert strong_morthogonal(H3, genus)

    print()
    print(shortstr((H2[:m, 1:n] + H3[:m, 1:n])%2))
    #print(eq2(H2[m+2:, i1+2:], H3[m:, i1:]))

    #print(classical_distance(H3))
    return H3


def glue_classical_self():

    from bruhat.triply_even import build

    genus = argv.get("genus", 3)

    m = argv.get("dim", 7)
    idx = argv.get("idx", 144)
    H = build.get(m, idx)
    H = H.astype(numpy.int32)

    count = argv.get("count", 1)
    for ii in range(count):
        m, n = H.shape
        R = rand2(m, m)
        H = dot2(R, H)
        if ii==0:
            print(H.shape)
            print(shortstrx(H))

        assert dot2(H, H.transpose()).sum() == 0
    
#        i0 = argv.get("i0", 0)
#        i1 = argv.get("i1", i0)
#        i2 = argv.get("i2", 1)
#        i3 = argv.get("i3", i2+1)

        items = list(range(n))
        i0 = random.choice(items)
        items.remove(i0)
        i1 = i0
        i2 = random.choice(items)
        items.remove(i2)
        i3 = random.choice(items)
        items.remove(i3)
        # glue i0<-->i2 and i1<-->i3
    
        assert strong_morthogonal(H, genus)
        print()
    
        H3 = glue_self_classical(H, [(i0, i2), (i1, i3)])
        print(H3.shape)
        print(shortstrx(H3))
        assert strong_morthogonal(H3, genus)
        H = H3
    
    return H


def glue_gcolor():
    from qupy.ldpc.gcolor import  Lattice
    l = argv.get('l', 1)
    lattice = Lattice(l)
    #code = lattice.build_code()
    H = lattice.Hx
    print("H:", H.shape)
    print(shortstr(H))
    m, n = H.shape
    H1 = zeros2(m+1, n+1)
    H1[1:, 1:] = H
    H1[0, :] = 1
#    print()
#    print(shortstr(H1))
#    for genus in range(1, 5):
#        print(genus, strong_morthogonal(H1, genus))

    H = H1

    genus = argv.get("genus", 3)

    H = H.astype(numpy.int32)
    n = H.shape[1]

    if argv.scramble:
        R = rand2(m, m)
        H = dot2(R, H)

    print("H:", H.shape)
    print(shortstrx(H))
    assert dot2(H, H.transpose()).sum() == 0

    i0 = argv.get("i0", 0)
    i1 = argv.get("i1", i0)
    i2 = argv.get("i2", n)
    i3 = argv.get("i3", i2+1)
    # glue i0<-->i2 and i1<-->i3

    H2 = direct_sum(H, H)
    print(H2.shape)
    print(shortstrx(H2))
    assert strong_morthogonal(H2, genus)
    print()

    H3 = glue_self_classical(H2, [(i0, i2), (i1, i3)])
    print(H3.shape)
    print(shortstrx(H3))
    assert strong_morthogonal(H3, genus)

    print()
    print(shortstr((H2[:m, 1:n] + H3[:m, 1:n])%2))
    #print(eq2(H2[m+2:, i1+2:], H3[m:, i1:]))

    #print(classical_distance(H3))
    return H3


if __name__ == "__main__":

    if argv.noerr:
        print("redirecting stderr to stderr.out")
        fd = os.open("stderr.out", os.O_CREAT|os.O_WRONLY)
        os.dup2(fd, 2)

    _seed = argv.get("seed")
    if _seed is not None:
        numpy.random.seed(_seed)
        random.seed(_seed)

    name = argv.next()
    if name is not None:
        f = eval(name)
        f()
    else:
        test()
        test_selfdual()
        test_colimit()
        test_equalizer()
        test_glue()
        test_color()
        test_universal()
        #test_ldpc()
        #test_quantum()

    print("OK")


