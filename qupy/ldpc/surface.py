#!/usr/bin/env python3

from collections import namedtuple
from functools import reduce
from operator import mul

import numpy
from numpy import concatenate as cat

from qupy.util import mulclose_fast
from qupy.argv import argv
from qupy.smap import SMap
from qupy.ldpc.solve import zeros2, shortstr, solve, dot2, array2, eq2, parse, pseudo_inverse
from qupy.ldpc.css import CSSCode
from qupy.ldpc.decoder import StarDynamicDistance


def mulclose_find(gen, names, target, verbose=False, maxsize=None):
    ops = set(gen)
    lookup = dict((g, (names[i],)) for (i, g) in enumerate(gen))
    bdy = gen
    dist = 1
    found = False
    while bdy:
        _bdy = set()
        for g in bdy:
            for h in gen:
                k = g*h
                if k in ops:
                    if len(lookup[g]+lookup[h]) < len(lookup[k]):
                        lookup[k] = lookup[g]+lookup[h]
                        assert 0
                    #continue

                else:
                    word = lookup[g]+lookup[h]
                    if len(word) > dist:
                        dist = len(word)
                        if verbose:
                            print("dist:", dist)
                        if found:
                            return
                    lookup[k] = word
                    ops.add(k)
                    _bdy.add(k)

                if k==target:
                    yield lookup[g]+lookup[h]
                    found = True
        bdy = _bdy
        #if verbose:
        #    print("mulclose:", len(ops))
        if maxsize and len(ops) >= maxsize:
            break


def mulclose_names(gen, names, verbose=False, maxsize=None):
    ops = set(gen)
    lookup = dict((g, (names[i],)) for (i, g) in enumerate(gen))
    bdy = gen
    while bdy:
        _bdy = set()
        for g in bdy:
            for h in gen:
                k = g*h
                if k in ops:
                    if len(lookup[g]+lookup[h]) < len(lookup[k]):
                        lookup[k] = lookup[g]+lookup[h]
                        assert 0
                else:
                    lookup[k] = lookup[g]+lookup[h]
                    ops.add(k)
                    _bdy.add(k)
        bdy = _bdy
        if verbose:
            print("mulclose:", len(ops))
        if maxsize and len(ops) >= maxsize:
            break
    return ops, lookup




class Clifford(object):
    def __init__(self, A):
        self.A = A
        m, n = A.shape
        self.shape = A.shape
        assert n%2 == 0
        self.n = n//2 # qubits
        self.key = A.tostring()

    def __str__(self):
        #s = str(self.A)
        #s = s.replace("0", ".")
        s = shortstr(self.A)
        return s

    def __mul__(self, other):
        assert isinstance(other, Clifford)
        A = dot2(self.A, other.A)
        return Clifford(A)

    def __call__(self, other):
        assert isinstance(other, CSSCode)
        assert other.n == self.n
        Lx, Lz, Hx, Tz, Hz, Tx, Gx, Gz = (
            other.Lx, other.Lz, other.Hx,
            other.Tz, other.Hz, other.Tx,
            other.Gx, other.Gz)
        assert Gx is None
        assert Gz is None
        A = self.A.transpose()
        LxLz = dot2(cat((Lx, Lz), axis=1), A)
        HxTz = dot2(cat((Hx, Tz), axis=1), A)
        TxHz = dot2(cat((Tx, Hz), axis=1), A)
        n = self.n
        Lx, Lz = LxLz[:, :n], LxLz[:, n:]
        Hx, Tz = HxTz[:, :n], HxTz[:, n:]
        Tx, Hz = TxHz[:, :n], TxHz[:, n:]
        code = CSSCode(Lx=Lx, Lz=Lz, Hx=Hx, Tz=Tz, Hz=Hz, Tx=Tx)
        return code

    def transpose(self):
        A = self.A.transpose().copy()
        return Clifford(A)

    def inverse(self):
        A = pseudo_inverse(self.A)
        return Clifford(A)

#    def __call__(self, other):
#        assert isinstance(other, CSSCode)
#        assert other.n == self.n
#        A = self.A.transpose()
#        B = other.to_symplectic()
#        C = dot2(B, A)
#        code = CSSCode.from_symplectic(C, other.n, other.k, other.mx, other.mz)
#        return code

    def __eq__(self, other):
        assert isinstance(other, Clifford)
        assert self.shape == other.shape
        #return eq2(self.A, other.A)
        return self.key == other.key

    def __ne__(self, other):
        return self.key != other.key

    def __hash__(self):
        return hash(self.key)

    @classmethod
    def identity(cls, n):
        A = zeros2(2*n, 2*n)
        for i in range(2*n):
            A[i, i] = 1
        return Clifford(A)

    @classmethod
    def hadamard(cls, n, idx):
        A = zeros2(2*n, 2*n)
        for i in range(2*n):
            if i==idx:
                A[i, n+i] = 1
            elif i==n+idx:
                A[i, i-n] = 1
            else:
                A[i, i] = 1
        return Clifford(A)

    @classmethod
    def cnot(cls, n, src, tgt):
        A = cls.identity(n).A
        assert src!=tgt
        A[tgt, src] = 1
        A[src+n, tgt+n] = 1
        return Clifford(A)
    
    @classmethod
    def swap(cls, n, idx, jdx):
        A = zeros2(2*n, 2*n)
        assert idx != jdx
        for i in range(n):
            if i==idx:
                A[i, jdx] = 1 # X sector
                A[i+n, jdx+n] = 1 # Z sector
            elif i==jdx:
                A[i, idx] = 1 # X sector
                A[i+n, idx+n] = 1 # Z sector
            else:
                A[i, i] = 1 # X sector
                A[i+n, i+n] = 1 # Z sector
        return Clifford(A)


class Surface(object):

    def __init__(self):
        self.keys = [] # list of coordinates (i, j, k)
        self.keymap = {} # map coordinate to index in keys
        #self.surf_keys = set() # surface code qubit coordinates
        #self.z_keys = set() # single qubit |0>  coordinates
        #self.x_keys = set() # single qubit |+>  coordinates
        self.tpmap = {} # map coordinate to tp "S", "Z", "X", "L"

    @property
    def n(self):
        return len(self.keys)

    def __repr__(self):
        return "Surface(%s)"%(self.keys,)
        
#    def __repr__(self):
#        surf_keys = self.surf_keys
#        z_keys = self.z_keys
#        x_keys = self.x_keys
#        return "Surface(%s, surf_keys=%s, z_keys=%s, x_keys=%s)" % (
#            self.keys, surf_keys, z_keys, x_keys)

    def get_keys(self, tp):
        assert tp in "SZXL"
        return [key for key in self.keys if self.tpmap[key] == tp]

    @property
    def surf_keys(self):
        return self.get_keys("S")

    @property
    def x_keys(self):
        return self.get_keys("X")

    @property
    def z_keys(self):
        return self.get_keys("Z")

    @property
    def logop_keys(self):
        return self.get_keys("L")
            
    def set_key(self, key, tp="L"):
        assert tp in ["S", "Z", "X", "L"]
        keys = self.keys
        keymap = self.keymap
        self.tpmap[key] = tp
        if key in keymap:
            return
        idx = len(keys)
        keys.append(key)
        keymap[key] = idx

    def del_key(self, key):
        keys = self.keys
        keymap = self.keymap
        tpmap = self.tpmap
        del tpmap[key]
        idx = keymap[key]
        del keymap[key]
        keys.pop(idx)
        for idx, key in enumerate(keys):
            keymap[key] = idx

    def set_many(self, top_left, bot_right, tp="S"):

        i0, j0 = top_left
        i1, j1 = bot_right

        # rough at top and bot
        # smooth at left and right
        for i in range(i0, i1):
          for j in range(j0, j1):
            ks = (0, 1)
            if i==i0:
                ks = (1,)
            if j==j1-1:
                ks = (1,)
            for k in ks:
                key = (i, j, k)
                self.set_key(key, tp)

    def del_many(self, top_left, bot_right):

        i0, j0 = top_left
        i1, j1 = bot_right

        # smooth at top and bot
        # rough at left and right
        for i in range(i0, i1):
          for j in range(j0, j1):
            ks = (0, 1)
            if i==i1-1:
                ks = (0,)
            if j==j0:
                ks = (0,)
            for k in ks:
                key = (i, j, k)
                self.del_key(key)

    def add_x(self, key):
        self.set_key(key, "X")

    def add_z(self, key):
        self.set_key(key, "Z")

    def add_logical(self, key):
        self.set_key(key, "L")

    def clone_keys(self):
        surf = Surface()
        for key in self.keys:
            surf.set_key(key)
        return surf

    def get_coord(self, i, j, k=None):
        if k is not None:
            assert k==0 or k==1
            row = 2*i + k
            col = 2*j + (1-k)
        else:
            row = 2*i
            col = 2*j
        return (row, col)

    def get_smap(self):
        smap = SMap()
        get_coord = self.get_coord
        for (i, j, k) in self.keys:
            smap[get_coord(i, j)] = "o"
            if k==0:
                smap[get_coord(i, j, k)] = "-"
            elif k==1:
                smap[get_coord(i, j, k)] = "|"
        return smap

    def __str__(self):
        smap = self.get_smap()
        return str(smap)

    def str_stab(self, stab, c="*"):
        smap = self.get_smap()
        for (i, j, k) in stab:
            smap[self.get_coord(i, j, k)] = c
        return str(smap)

    def str_op(self, op, c="*"):
        smap = self.get_smap()
        for idx, key in enumerate(self.keys):
            if op[idx]==0:
                continue
            (i, j, k) = key
            smap[self.get_coord(i, j, k)] = c
        return str(smap)

    def get_stabs(self):
        keys = self.keys
        keymap = self.keymap
        surf_keys = self.surf_keys
        x_keys = self.x_keys
        z_keys = self.z_keys
        i0 = min(i for (i,j,k) in keys)
        i1 = max(i for (i,j,k) in keys)
        j0 = min(j for (i,j,k) in keys)
        j1 = max(j for (i,j,k) in keys)

        z_ops = []
        deltas = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 0, 0)]
        for i in range(i0, i1+1):
          for j in range(j0, j1+1):
            stab = []
            for di, dj, k in deltas:
                key = (i+di, j+dj, k)
                if key not in surf_keys:
                    continue
                stab.append(key)
            if len(stab)>=3:
                z_ops.append(stab)

        x_ops = []
        deltas = [(0, 0, 0), (0, 0, 1), (-1, 0, 1), (0, -1, 0)]
        for i in range(i0, i1+1):
          for j in range(j0, j1+1):
            stab = []
            for di, dj, k in deltas:
                key = (i+di, j+dj, k)
                if key not in surf_keys:
                    continue
                stab.append(key)
            if len(stab)>=3:
                x_ops.append(stab)

        for key in keys:
            if key in x_keys:
                x_ops.append([key])
        for key in keys:
            if key in z_keys:
                z_ops.append([key])

        return x_ops, z_ops

    def check(self):
        x_ops, z_ops = self.get_stabs()
        for xop in x_ops:
          for zop in z_ops:
            xop = set(xop)
            zop = set(zop)
            n = len(xop.intersection(zop))
            if n%2:
                print("FAIL "*10)
                print(self.str_stab(xop, "X"))
                print("FAIL "*10)
                print(self.str_stab(zop, "Z"))
                assert 0
                return

    def get_code(self):
        self.check()
        keys = self.keys
        keymap = self.keymap
        x_ops, z_ops = self.get_stabs()
        n = len(keys)
        mx = len(x_ops)
        mz = len(z_ops)
        Hx = zeros2(mx, n)
        Hz = zeros2(mz, n)
        for idx, stab in enumerate(x_ops):
            for key in stab:
                Hx[idx, keymap[key]] = 1
        for idx, stab in enumerate(z_ops):
            for key in stab:
                Hz[idx, keymap[key]] = 1
        code = CSSCode(Hx=Hx, Hz=Hz)
        return code


    def dump(self):
        sep = "_"*self.n*2
        for tp in "SXZL":
            smap = self.get_smap()
            for key in self.get_keys(tp):
                (i, j, k) = key
                smap[self.get_coord(i, j, k)] = tp
            print(sep)
            print("tp = %r"%(tp,))
            print(smap)
        print(sep)

    def dump_idxs(self):
        n = self.n
        keys = self.keys
        sep = '.'*2*self.n
        for idx in range(n):
            smap = self.get_smap()
            (i, j, k) = keys[idx]
            smap[self.get_coord(i, j, k)] = str(idx)
            print(smap)
            print(sep)

    def dump_stabs(self):
        x_ops, z_ops = self.get_stabs()
        for stab in x_ops:
            print()
            print(self.str_stab(stab, "X"))
        for stab in z_ops:
            print()
            print(self.str_stab(stab, "Z"))

    def dump_code(self, code):
        assert code.n == self.n
        str_op = self.str_op
        w = 2*self.n
        print("Lx:")
        for op in code.Lx:
            print(str_op(op, "X"))
            print("."*w)
        print("Lz:")
        for op in code.Lz:
            print(str_op(op, "Z"))
            print("."*w)
        print("Hx:")
        for op in code.Hx:
            print(str_op(op, "X"))
            print("."*w)
        print("Hz:")
        for op in code.Hz:
            print(str_op(op, "Z"))
            print("."*w)


def get_gen(n, pairs=None):
    #gen = [Clifford.hadamard(n, i) for i in range(n)]
    #names = ["H_%d"%i for i in range(n)]
    gen = []
    names = []
    if pairs is None:
        pairs = []
        for i in range(n):
          for j in range(n):
            if i!=j:
                pairs.append((i, j))
    for (i, j) in pairs:
        assert i!=j
        gen.append(Clifford.cnot(n, i, j))
        names.append("CN(%d,%d)"%(i,j))

    return gen, names


def test_symplectic():

    n = 3
    I = Clifford.identity(n)
    for idx in range(n):
      for jdx in range(n):
        if idx==jdx:
            continue
        CN_01 = Clifford.cnot(n, idx, jdx)
        CN_10 = Clifford.cnot(n, jdx, idx)
        assert CN_01*CN_01 == I
        assert CN_10*CN_10 == I
        lhs = CN_10 * CN_01 * CN_10
        rhs = Clifford.swap(n, idx, jdx)
        assert lhs == rhs
        lhs = CN_01 * CN_10 * CN_01
        assert lhs == rhs

    n = 4
    I = Clifford.identity(n)
    H = Clifford.hadamard(n, 0)
    assert H*H == I

    CN_01 = Clifford.cnot(n, 0, 1)
    assert CN_01*CN_01 == I

    n = 3
    trivial = CSSCode(
        Lx=parse("1.."), Lz=parse("1.."), Hx=zeros2(0, n), Hz=parse(".1. ..1"))

    assert trivial.row_equal(CSSCode.get_trivial(3, 0))

    repitition = CSSCode(
        Lx=parse("111"), Lz=parse("1.."), Hx=zeros2(0, n), Hz=parse("11. .11"))

    assert not trivial.row_equal(repitition)

    CN_01 = Clifford.cnot(n, 0, 1)
    CN_12 = Clifford.cnot(n, 1, 2)
    CN_21 = Clifford.cnot(n, 2, 1)
    CN_10 = Clifford.cnot(n, 1, 0)
    encode = CN_12 * CN_01

    code = CN_01 ( trivial )
    assert not code.row_equal(repitition)
    code = CN_12 ( code )
    assert code.row_equal(repitition)

#    assert (CN_21 * CN_10)(trivial).row_equal(repitition)
#    assert (CN_10 * CN_21)(trivial).row_equal(repitition)

#    src = Clifford(trivial.to_symplectic())
#    src_inv = src.inverse()
#    tgt = Clifford(repitition.to_symplectic())
#    A = (src_inv * tgt).transpose()
#    #print(A)

    A = get_encoder(trivial, repitition)

    gen, names = get_gen(3)
    for word in mulclose_find(gen, names, A):
        #print("word:")
        #print(word)
    
        items = [gen[names.index(op)] for op in word]
        op = reduce(mul, items)
    
        #print(op)
        #assert op*(src) == (tgt)
    
        #print(op(trivial).longstr())
        assert op(trivial).row_equal(repitition)
    

def get_encoder(source, target):
    assert isinstance(source, CSSCode)
    assert isinstance(target, CSSCode)
    src = Clifford(source.to_symplectic())
    src_inv = src.inverse()
    tgt = Clifford(target.to_symplectic())
    A = (src_inv * tgt).transpose()
    return A


def test_encode():

    surf = Surface()
    surf.set_many((0, 0), (2, 2))
    #print(surf)

    target = surf.get_code()
    n = target.n
    assert n==5

    #idx = surf.keymap[(1, 0, 0)]
    #source = CSSCode.get_trivial(surf.n, idx)
    #print(source.longstr())

    trivial = surf.clone_keys()
    trivial.add_x(trivial.keys[0])
    trivial.add_x(trivial.keys[1])
    trivial.add_z(trivial.keys[2])
    trivial.add_z(trivial.keys[3])
    
    source = trivial.get_code()
    #print("source:")
    #print(source.longstr())
    #print("target")
    #print(target.longstr())

    if 0:
        op = Clifford.identity(n)
        #for i in [0, 1, 3, 4]:
        for i in [0, 4]:
            op = op*Clifford.hadamard(n, i)
        source = op(source)

    A = get_encoder(source, target)

    pairs = None
    #pairs = [(0, 2), (1, 2), (3, 2), (4, 2)]
    #pairs = [(0, 2), (1, 2), (3, 2), (4, 2), (2, 0), (2, 1), (2, 3), (2, 4), (1, 4), (4, 3)] #... etc....
    pairs = [(0, 3), (1, 0), (1, 2), (1, 4), (2, 3), (4, 3)]
    pairs += [(0, 2)]
    gen, names = get_gen(surf.n, pairs)
    #G = mulclose_fast(gen)

    B = None
    for result in mulclose_find(gen, names, A):
        #print(len(result), ":", "*".join(result))
        B = reduce(mul, [gen[names.index(op)] for op in result])
        #print(B)
        break

    assert B is not None
    assert B==A
    #print("result")
    code = B(source)
    #print(code.longstr())
    assert code.row_equal(target)


def test_surface():

    """
    Try to encode n=5 surface code into n=25 surface code
    """

    source = Surface()
    source.set_many((0, 0), (4, 4), "L")
    source.set_many((1, 1), (3, 3))

    source.set_key((0, 1, 1), "X")
    source.set_key((0, 2, 1), "X")
    source.set_key((3, 1, 1), "X")
    source.set_key((3, 2, 1), "X")

    source.set_key((1, 0, 0), "X")
    source.set_key((2, 0, 0), "X")
    source.set_key((3, 0, 0), "X")
    source.set_key((1, 2, 0), "X")
    source.set_key((2, 2, 0), "X")
    source.set_key((3, 2, 0), "X")

    for key in source.keys:
        if source.tpmap[key] == "L":
            source.set_key(key, "Z")

    assert source.n == 25

    #print(repr(source))
    source.dump()
    print(source.get_code().longstr())
    print()

    target = Surface()
    target.set_many((0, 0), (4, 4))
    print(target)
    print(target.get_code().longstr())

    A = get_encoder(source.get_code(), target.get_code())
    #print(A)


def test_puncture():

    source = Surface()
    source.set_many((0, 0), (7, 7), "S")
    source.del_many((2, 1), (6, 5))
    source.set_key((2, 1, 0), "S")
    source.set_key((5, 1, 0), "S")
    source.set_key((2, 4, 0), "S")
    source.set_key((5, 4, 0), "S")
    print(source.get_smap())

    source.check()
    #source.dump_stabs()

    code = source.get_code()
    print(code)

    #source.dump_code(code)

    decoder = StarDynamicDistance(code)

    for idx in range(4):
        op = decoder.find(idx=idx)
        print(source.str_op(op))
        print()



if __name__ == "__main__":

    name = argv.next()
    if name:

        fn = eval(name)
        fn()

    else:
    
        test_symplectic()
        test_encode()
        test_surface()
        test_puncture()




