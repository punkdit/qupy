#!/usr/bin/env python3

import sys

import numpy
from numpy import dot

from qupy.ldpc import solve
from qupy.ldpc.solve import parse, shortstr, array2, zeros2, eq2, compose2
from qupy.ldpc import solve
from qupy.ldpc.tool import write


def zeros(shape):
    a = numpy.zeros(shape, dtype=numpy.int32)
    return a


def tensor(A, B):
    C = numpy.tensordot(A, B, 0)
    C = C.transpose([0, 2, 1, 3])
    C = C.copy()
    C.shape = (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1])
    return C


def identity(n):
    I = numpy.eye(n, dtype=numpy.int32)
    return I


class BlockArray(object):
    """ A1*B1 + A2*B2 + ... -> C1*D1 + ...
        ^^^^^^^^ cols ^^^^^ -> ^^^^^^^ rows ^^^
    """

    def __init__(self):
        #self.rows = []
        #self.cols = []
        #print "BlockArray"
        self.data = {} # map (row, col) -> array

    def add(self, row, col, H):
        assert H.shape[0]*H.shape[1], H.shape
        #print "BlockArray", row, col, H.shape
        data = self.data
        H0 = data.get((row, col))
        if H0 is None:
            data[row, col] = H.copy() # <-- copy
        else:
            assert H.shape == H0.shape, (H.shape, H0.shape)
            H0 += H

    def todense(self, cols=None):
        data = self.data
        #keys = data.keys()
        #rows = [row for row, col in keys]
        #cols = [col for row, col in keys]
        shape = [0, 0]
#        print "todense", data.keys()
        for (row, col), H in list(data.items()):
            r = row + H.shape[0]
            c = col + H.shape[1]
            shape[0] = max(r, shape[0])
            shape[1] = max(c, shape[1])
        if cols is not None:
            assert cols >= shape[1]
            shape = shape[0], cols
        H0 = zeros(shape)
#        print shortstr(H0)
        for (row, col), H in list(data.items()):
#            print row, col
#            print shortstr(H)
            H0[row : row+H.shape[0], col : col+H.shape[1]] += H
        return H0


class Chain(object):

    """
    Chain Complex:

        Hs[0]   Hs[1]   Hs[2]   Hs[3]
      . <---- . <---- . <---- . <----      etc
     C_-1     C_0     C_1     C_2     C_3

    Hs[i].shape[1] == Hs[i+1].shape[0]
    """

    def __init__(self, Hs, Ls=None, d=2, dual=None, check=True):
        self.d = d
        #self.Hs = [H for H in Hs if H.shape[0] or H.shape[1]]
        self.Hs = list(Hs)
        self.Ls = Ls # homology
        self._dual = dual
        if check:
            self.check()

    def check(self):
        #print "check"
        d = self.d
        Hs = self.Hs
        i = 0
        while i+1 < len(Hs):
            A, B = Hs[i:i+2]
            #if B.shape[0]*B.shape[1]==0:
            #    break # XX fixy fixy XX
            assert A.shape[1] == B.shape[0], (A.shape, B.shape)
            #print shortstr(A)
            #print
            #print shortstr(B.transpose())
            C = dot(A, B)
            #print
            #print shortstr(C)
            C %= d
            assert C.sum() == 0
            #print "OK",
            i += 1

    def homology(self, i):
        "kern of Hs[i] mod image Hs[i+1]"
        H0, H1 = self[i], self[i+1]
        #print "homology", i
        #print H0.shape, H1.shape
        L = solve.find_logops(H0, H1.transpose())
        return L

    def __getitem__(self, idx):
        # Hs[idx] : C_idx -> C_{idx-1}
        # shape = dim(idx-1), dim(idx)
        Hs = self.Hs
        if idx<0 or idx>=len(Hs):
            shape = self.dim(idx-1), self.dim(idx)
            H = zeros2(*shape)
        else:
            H = self.Hs[idx]
        return H

    def __len__(self):
        return len(self.Hs)

    def __eq__(self, other):
        if self is other:
            return True
        assert 0, "not implemented"

    def __ne__(self, other):
        assert 0, "not implemented"

    def to_zero(self):
        Hs = self.Hs
        Ds = []
        chain = [] # _zero
        n = len(Hs)
        for i in range(n):
            H = Hs[i]
            chain.append(zeros2(0, 0))
            Ds.append(zeros2(0, H.shape[0]))
        Ds.append(zeros2(0, H.shape[1]))
        chain = Chain(chain)
        assert len(self) == len(chain)
        f = Map(self, chain, Ds)
        return f

    def from_zero(self):
        dual = self.get_dual()
        f = dual.to_zero()
        f = f.get_dual()
        assert f.src is self
        return f

    def get_dual(self):
        if self._dual is None:
            Hs = self.Hs
            Hs = list(reversed([H.transpose() for H in Hs]))
            self._dual = Chain(Hs, d=self.d, dual=self)
        return self._dual

    def dim(self, i):
        Hs = self.Hs
        if i>=len(Hs):
            m = 0
        elif i>=0:
            m = Hs[i].shape[1]
        elif i==-1:
            m = Hs[0].shape[0]
        else:
            m = 0
        return m

    # Call chains X, Y ? or other letters: C, D ?

    def tensor(X, Y, idx=None):
        assert X.d == Y.d
        if idx is None:
            # recurse
            n = max(len(X), len(Y))+2
            items = [X.tensor(Y, idx) for idx in range(n)]
            Hs = [item[0] for item in items]
            Ls = [item[1] for item in items]
            chain = Chain(Hs, Ls, d=X.d)
            return chain

        assert idx>=0

        n = 0
        coldims = [] # src
        for i in range(-1, idx+1):
            j = idx-i-1
            coldims.append(n)
            n += X.dim(i) * Y.dim(j)
        coldims.append(n)

        n = 0
        rowdims = [] # tgt
        for i in range(-1, idx):
            j = idx-i-2
            rowdims.append(n)
            n += X.dim(i) * Y.dim(j)
        rowdims.append(n)

        #print "tensor"
        #print '\t', idx, coldims, rowdims

        d = X.d
        H = BlockArray()
        L = BlockArray()
        row = 0
        for i in range(-1, idx+1):
            j = idx-i-1

            """
            X[i]*I : X.dim(i)*Y.dim(j) -> X.dim(i-1)*Y.dim(j)
            I*Y[j] : X.dim(i)*Y.dim(j) -> X.dim(i)*Y.dim(j-1)
            """

            XL = X.homology(i)
            YL = Y.homology(j)
            A = tensor(XL, YL)
            #print '\t', i, j, XL.shape, YL.shape, A.shape
            if A.shape[0]*A.shape[1]:
                L.add(row, coldims[i+1], A)
                row += A.shape[0]

            if 0<=i<len(X) and Y.dim(j):
                I = identity(Y.dim(j))
                # (i, j), (i-1, j)
                #print "tensor:", X[i].shape, I.shape
                A = tensor(X[i], I)
                H.add(rowdims[i], coldims[i+1], A)

            if 0<=j<len(Y) and X.dim(i):
                I = identity(X.dim(i))
                # (i, j), (i, j-1)
                B = tensor(I, Y[j])
                H.add(rowdims[i+1], coldims[i+1], B)
            
        H = H.todense()
        L = L.todense(cols=coldims[-1])
        #print "H:", H.shape
        #print "L:", L.shape
#        if L.shape[0] == 0:
#            L.shape = 0, H.shape[1]
#        assert L.shape[1] == H.shape[1], (L.shape, H.shape)
#        if L.shape[0]*L.shape[1]==0:
#            L = None # XXX Fix my shape :-(
        return H, L

    def get_code(self, idx=0, build=True, check=True):
        #print "get_code"
        Hs = self.Hs
        Hx = Hs[idx]
        write("li:")
        Hx = solve.linear_independent(Hx)
        Hz = Hs[idx+1].transpose()
        write("li:")
        Hz = solve.linear_independent(Hz)
        #print(Hx.shape, Hz.shape)
        if self.Ls:
            Lz = self.Ls[idx]
            #print "get_code:", Lz.shape
        else:
            Lz = None
        #print shortstr(dot(Hx, Hz.transpose()))
        #print "Hz:"
        #print shortstr(Hz)
        #print "Hx:"
        #print shortstr(Hx)
        from qupy.ldpc.css import CSSCode
        code = CSSCode(Hx=Hx, Hz=Hz, Lz=Lz, build=build, check=check)
        return code

    def allcodes(self, **kw):
        for i in range(len(self)-1):
            code = self.get_code(i, **kw)
            yield code

    def dumpcodes(self, **kw):
        for code in self.allcodes(**kw):
            print(code)
            #print code.weightsummary()
            print(code.weightstr())


class Map(object):
    def __init__(self, src, tgt, Ds, check=True):
        assert isinstance(src, Chain)
        assert isinstance(tgt, Chain)
        assert len(src) == len(tgt)
        assert len(src) == len(Ds)-1
        assert src.d == tgt.d
        self.d = src.d
        self.src = src
        self.tgt = tgt
        self.shape = (tgt, src)
        Ds = list(Ds)
        self.Ds = Ds
        for i, D in enumerate(Ds):
            assert D.shape == (tgt.dim(i-1), src.dim(i-1)), "i = %d"%i
        if check:
            self.check()

    def __len__(self):
        return len(self.Ds)

    def __getitem__(self, idx):
        return self.Ds[idx]

    def __add__(self, other):
        assert len(self) == len(other)
        assert self.shape == other.shape
        n = len(self)
        d = self.d
        Ds = [(self[i]+other[i])%d for i in range(n)]
        return Map(self.src, self.tgt, Ds)

    def __sub__(self, other):
        assert len(self) == len(other)
        assert self.shape == other.shape
        n = len(self)
        d = self.d
        Ds = [(self[i]-other[i])%d for i in range(n)]
        return Map(self.src, self.tgt, Ds)

    def check(self):
        src, tgt = self.src, self.tgt
        Ds = self.Ds
        for i in range(len(src)):
            D0, D1 = Ds[i], Ds[i+1]
            if D0 is None or D1 is None:
                continue
            lhs, rhs = compose2(src[i], D0), compose2(D1, tgt[i])
            if not eq2(lhs, rhs):
                print("Not a chain map:")
                print("lhs =", lhs)
                print("rhs =", rhs)
                raise Exception


def pushout(a, b):
    assert isinstance(a, Map)
    assert isinstance(b, Map)
    assert a.src == b.src

    src = a.src
    n = len(src)
    amap = []
    bmap = []
    chain = []
    aprev = None
    bprev = None
    for i in range(n+1):
        a1, b1, c1 = solve.pushout(a[i], b[i], aprev, bprev)
        amap.append(a1)
        bmap.append(b1)
        chain.append(c1)
        aprev = compose2(a.tgt[i], a1)
        bprev = compose2(b.tgt[i], b1)
    c = Chain(chain[1:])
    amap = Map(a.tgt, c, amap)
    bmap = Map(b.tgt, c, bmap)
    return amap, bmap, c


def equalizer(a, b):
    assert isinstance(a, Map)
    assert isinstance(b, Map)
    assert a.shape == b.shape
    tgt, src = a.shape

    ab = a-b
    c = src.to_zero()
    amap, bmap, chain = pushout(ab, c)
    return amap, bmap, chain


def test_repitition():

    H = parse(
    """
        111
        11.
        .11
    """)

    X = Chain([H])
    X.check()

    XX = X.tensor(X.get_dual(), 1)
    print(XX)
    print(shortstr(XX[0]))


def test_stean():

    Hx = parse(
    """
        ...1111
        .11..11
        1.1.1.1
        1.11.1.
        .1111..
    """)

    Hx = parse(
    """
        ...1111
        .11..11
        1.1.1.1
    """)

    Hz = parse(
    """
        ...1111
        .11..11
        1.1.1.1
    """)

    A = list(a for a in solve.span(Hz) if a.sum())
    A = array2(A)
    print(shortstr(A))
    Hx = A

    Hz = Hx.copy()

    X = Chain([Hx, Hz.transpose()])

    X.check()

#    for i in range(-1, 3):
#        print "homology", i
#        print shortstr(X.homology(i))

#    X.dumpcodes()
#    print
#    print "=========="*10
#    print

    XX = X.tensor(X)

    XX.dumpcodes()
    return

    for L in XX.Ls:
        print(L.shape)

    code = XX.get_code(1)
    print(code)
    print(code.weightsummary())
    #code.save("stean2_147_33.ldpc")
    return

    #XX = XX.tensor(X)

    XX.dumpcodes()

    return

    #code = XX.get_code(1)

    #code.save('stean2.ldpc')

    X4 = XX.tensor(XX)

    X4.dumpcodes()


def test_hyperbicycle():

    name = sys.argv[1]

    from qupy.ldpc.css import CSSCode
    code = CSSCode.load(name)

    X = Chain([code.Hx, code.Hz.transpose()])
    X.check()

    # 0 -> 1 -> 2 -> 3 -> 4
    #XX = X.tensor(X, 1) # very big :-(

    XX = X.tensor(X)
    #XX.dumpcodes()

    code = XX.get_code(1)
    print(code)

    code.save("hyper_%d_%d.ldpc"%(code.n, len(code.Lx)))

#test_hyperbicycle()
#sys.exit(0)




def test_toric4():

    d = 2

    H = parse(
    """
    1100
    0110
    0011
    1001
    """)

    T1 = Chain([H])
    T1t = T1.get_dual()

    T2 = T1.tensor(T1t)
    assert len(T2)==2
    T2.check()

    #T3 = T2.tensor(T1)
    T4 = T2.tensor(T2)

    code = T4.get_code(1)
    print(code)

    code.save("toric4_%d_%d.ldpc"%(code.n, len(code.Lx)))


def test_product():

    from qupy.ldpc import css

    if len(sys.argv) < 2:
        return

    code = css.CSSCode.load(sys.argv[1])

    if not code.mx:

        X1 = Chain([code.Hz])
        X2 = Chain([code.Hz])
    
        XX = X1.tensor(X2.get_dual())
        #XX = X1.tensor(X2)
    
        print(len(XX.Hs))
    
        code = XX.get_code(0)

    else:

        X1 = Chain([code.Hx, code.Hz.transpose()])
        X1.check()
        #print X1.get_code(0)

        X2 = Chain([code.Hx, code.Hz.transpose()])

        XX = X1.tensor(X2)

        code = XX.get_code(1)

        print(code)

        

    #Hx, Hzt = XX.Hs

    #code = css.CSSCode(Hx=Hx, Hz=Hzt.transpose(), build=False, check=False)

    #print code.longstr()

    if code.k:
        code.save(stem="prod")



if __name__ == "__main__":

    test_repitition()
    test_stean()
    test_product()


