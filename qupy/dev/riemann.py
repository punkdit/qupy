#!/usr/bin/env python3

from time import sleep
from functools import reduce
from operator import matmul

from qupy.dev._algebra import Algebra, build_algebra, Tensor
#from qupy.util import mulclose # slooooow
from qupy.ldpc.solve import remove_dependent, parse, shortstr, eq2
from qupy.argv import argv

#from qupy.dev.su2 import StabilizerCode

# Real pauli group
#pauli = build_algebra("IXZY", "X*X=I Z*Z=I Y*Y=-I X*Z=Y Z*X=-Y X*Y=Z Y*X=-Z Z*Y=-X Y*Z=X")

# _Complex pauli group
pauli = build_algebra("IXZY",
    "I*I=I I*X=X I*Z=Z I*Y=Y X*I=X X*X=I X*Z=-1i*Y"
    " X*Y=1i*Z Z*I=Z Z*X=1i*Y Z*Z=I Z*Y=-1i*X Y*I=Y Y*X=-1i*Z Y*Z=1i*X Y*Y=I")

I = pauli.I
X = pauli.X
Y = pauli.Y
Z = pauli.Z

S = (1+1j)/2 * I + (1-1j)/2 * Z
Si = (1-1j)/2 * I + (1+1j)/2 * Z
assert S*S == Z
assert Si*S == I
assert S*Si == I
assert Si*Si == Z

def make_op(opi, i, n):
    ops = [I]*n
    ops[i] = opi
    op = reduce(matmul, ops)
    return op

def make_cz(src, tgt, n):
    assert src != tgt
    assert 0 <= src < n
    assert 0 <= tgt < n
    desc = ["I"]*n
    op  = pauli.parse(desc) # II
    desc[src] = "Z"
    op += pauli.parse(desc) # ZI
    desc[tgt] = "Z"
    op -= pauli.parse(desc) # ZZ
    desc[src] = "I"
    op += pauli.parse(desc) # IZ
    op = (1/2)*op
    return op

def make_cx(src, tgt, n):
    assert src != tgt
    assert 0 <= src < n
    assert 0 <= tgt < n
    desc = ["I"]*n
    op  = pauli.parse(desc) # II
    desc[src] = "X"
    op += pauli.parse(desc) # XI
    desc[tgt] = "Z"
    op -= pauli.parse(desc) # XZ
    desc[src] = "I"
    op += pauli.parse(desc) # IZ
    op = (1/2)*op
    return op

def make_swap(src, tgt, n):
    return make_cx(src, tgt, n) * make_cx(tgt, src, n)

CX = make_cx(0, 1, 2)
assert CX*CX == I@I


def mk_op(desc, op):
    assert op in "XZY"
    desc = desc.replace(".", "I")
    desc = desc.replace("1", op)
    op = pauli.parse(desc)
    return op

def mk_zop(descs):
    ops = [mk_op(desc, "Z") for desc in descs.split()]
    return ops

def mk_xop(descs):
    ops = [mk_op(desc, "X") for desc in descs.split()]
    return ops




class Code(object):
    def __init__(self, Hxf, Hzf):
        Hx = remove_dependent(Hxf)
        Hz = remove_dependent(Hzf)

        mx, n = Hx.shape
        mz, n = Hz.shape
    
        xstabs = mk_xop(shortstr(Hx))
        zstabs = mk_zop(shortstr(Hz))
    
        for zop in zstabs:
          for xop in xstabs:
            assert zop*xop == xop*zop
    
        I = pauli.parse("I"*n)
    
        Pz = I
        for op in zstabs:
            Pz = (0.5)*Pz*(I + op)
    
        Px = I
        for op in xstabs:
            Px = (0.5)*Px*(I + op)

        self.Hxf = Hxf
        self.Hzf = Hzf
        self.Hx = Hx
        self.Hz = Hz
        self.Px = Px
        self.Pz = Pz
        self.n = n
        self.mx = mx
        self.mz = mz
        self.k = n-mx-mz

    def __str__(self):
        return "Code(n=%d, mx=%d, mz=%d, k=%d)"%(
            self.n, self.mx, self.mz, self.k)

    def __eq__(self, other):
        return eq2(self.Hx, other.Hx) and eq2(self.Hz, other.Hz)

    def get_projector(self):
        P = self.Px*self.Pz # Ok, uses 8.3 GB ram
        return P

    def get_perm(self, perm):
        Hx, Hz = self.Hx, self.Hz
        Hx = Hx[:, perm]
        Hz = Hz[:, perm]
        code = Code(Hx, Hz)
        return code

    def get_dual(self):
        Hx, Hz = self.Hx, self.Hz
        return Code(Hz, Hx)


def main():

    Hz = parse("""
    11111.........................
    1....1111.....................
    .1.......1111.................
    ..1..1.......111..............
    ...1..1..1......11............
    ....1.....1..1....11..........
    ...........1....1...111.......
    ............1.....1.1..11.....
    ..............1....1...1.11...
    .......1.......1.........1.11.
    ........1........1...1.....1.1
    ......................1.1.1.11
    """)

    Hx = parse("""
    11......1..1.........1........
    .....11..11..1................
    ...11...........1.1.1.........
    .........1..1....1......1....1
    ....................11.1.1.1..
    ..........11.......1..1...1...
    .............1.1..1.....1...1.
    ..11...........1.1.........1..
    .....1..1.....1...........1..1
    .11.........1.1........1......
    1...1..1...........1.....1....
    ......11........1.....1.....1.
    """)

    mx, n = Hx.shape
    mz, n = Hz.shape

    from qupy.condmat.isomorph import Tanner, search

    src = Tanner.build2(Hx, Hz)

    #tgt = Tanner.build2(Hx, Hz)
    tgt = Tanner.build2(Hz, Hx) # weak duality

    fns = []
    for fn in search(src, tgt):
        assert len(fn) == mx+mz+n
        bitmap = []
        for i in range(n):
            bitmap.append( fn[i+mx+mz]-mx-mz )
        #print(bitmap)
        fixed = [i for i in range(n) if bitmap[i]==i]
        if len(fixed)==2:
            #print(bitmap)
            #print(len(fixed), end=" ")
            break

    perm = tuple(bitmap)
    print(perm)

    # XX too big...
#    fold = I
#    for i, j in enumerate(perm):
#        if i < j:
#            print("swap", i, j)
#            op = make_swap(i, j, n)
#            fold = op*fold

    code = Code(Hx, Hz)
    print(code)

    code1 = code.get_perm(perm)
    print(code1)
    assert not code==code1

    In = pauli.parse("I"*n)
    fold = In
    for i, j in enumerate(perm):
        if i < j:
            fold *= make_cz(i, j, code.n)
        elif i==j:
            fold *= make_op(S, i, code.n)

    print("fold")

    #lhs = fold * code.get_projector()
    lhs = fold * code.Pz * code.Px
    print("lhs")
    rhs = code1.Pz * (code1.Px * fold)
    print("rhs")
    print(lhs == rhs)

    if argv.pause:
        print("done...")
        while 1:
            sleep(1)
    

if __name__ == "__main__":

    main()

    print("OK")

