#!/usr/bin/env python3

"""
look for 4-qubit codes with transverse S gate.
"""


from functools import reduce
from operator import matmul, add

from qupy.dev.su2 import StabilizerCode, cyclotomic
#from qupy.dev._algebra import Algebra, build_algebra, Tensor
from qupy.dev.algebra import Algebra, build_algebra, Tensor
from qupy.dense import Gate, Qu
from qupy.tool import cross
from qupy.util import mulclose
from qupy.argv import argv


def test():
    
    I = Gate.I
    X = Gate.X
    Z = Gate.Z
    Y = Gate.Y
    
    i = 1.j
    assert X*Z == -i*Y
    assert Z*X == i*Y
    assert X*Y == i*Z
    assert Y*X == -i*Z
    assert Z*Y == -i*X
    assert Y*Z == i*X

    S = Gate.S
    Sd = S.dag()

    assert S*Sd == I
    assert S*Z*Sd == Z
    assert S*X*Sd == Y
    assert S*Y*Sd == -X

    assert str(X*Z) == "-1.0j*Y"


def get_dense(op):
    
    I = Gate.I
    X = Gate.X
    Z = Gate.Z
    Y = Gate.Y

    return op.evaluate({"I":I, "X":X, "Z":Z, "Y":Y})
    

def get_group(n):
    
    I = Gate.I
    X = Gate.X
    Z = Gate.Z
    Y = Gate.Y

    S = Qu((2, 2), 'ud', [[1, 0], [0, cyclotomic(4)]])

    for op in [X, Z]:
        op = reduce(matmul, [op]*n)
        yield op


def all_codes_3(pauli, n):
    I = pauli.I
    X = pauli.X
    Y = pauli.Y
    Z = pauli.Z

    In = reduce(matmul, [I]*n)
    all_ops = [reduce(matmul, op) for op in cross([[I, X, Y, Z]]*n)]
    assert all_ops[0] == In
    all_ops.pop(0)

    N = len(all_ops)
    print("all_ops:", N)

    for i in range(N):
      opi = all_ops[i]
      for j in range(i+1, N):
        opj = all_ops[j]
        if opi*opj != opj*opi:
            continue
        for k in range(j+1, N):
            opk = all_ops[k]
            if opi*opk != opk*opi:
                continue
            if opj*opk != opk*opj:
                continue
            yield [opi, opj, opk]
    

def uniq_codes(pauli, n):

    I = pauli.I
    X = pauli.X
    Y = pauli.Y
    Z = pauli.Z

    In = reduce(matmul, [I]*n)

    count = 0
    ops = {}
    for gen in all_codes_3(pauli, n):
        G = mulclose(gen)
        if len(G) != 8:
            continue
        if -In in G:
            continue
        count += 1
        P = reduce(add, G)
        s = str(P)
        if s not in ops:
            ops[s] = P
            yield P

    print("codes:", count)
    print("uniq:", len(ops))



def main():

    #pauli = build_algebra("IXZY", "X*X=I Z*Z=I Y*Y=-I X*Z=Y Z*X=-Y X*Y=Z Y*X=-Z Z*Y=-X Y*Z=X")
    pauli = build_algebra("IXZY", "X*X=I Z*Z=I Y*Y=I X*Z=-iY Z*X=iY X*Y=iZ Y*X=-iZ Z*Y=-iX Y*Z=iX")

    I = pauli.I
    X = pauli.X
    Y = pauli.Y
    Z = pauli.Z

    n = argv.get("n", 4)

    #all_ops = [reduce(matmul, op) for op in cross([[I, X, Y, Z]]*n)]

    if 0:
        code = StabilizerCode(pauli, "XXXI ZZIZ IYYY")
        #code = StabilizerCode(pauli, "XYZI IXYZ ZIXY")
        #code = StabilizerCode(pauli, "XZZXI IXZZX XIXZZ ZXIXZ")
        n = code.n
    
        P = code.get_projector()
        print(P)


    I = Gate.I
    X = Gate.X
    Z = Gate.Z
    Y = Gate.Y
    S = Gate.S
    Sd = S.dag()
    ns = {"I":I, "X":X, "Z":Z, "Y":Y, "S":S}

    Xt = reduce(matmul, [X]*n)
    Zt = reduce(matmul, [Z]*n)
    Yt = reduce(matmul, [Y]*n)
    St = reduce(matmul, [S]*n)

    if 0:
        for g in all_ops:
            desc = str(g)
            g = get_dense(g)
            if g*St != St*g:
                continue

                print(desc)
        return

    #code = StabilizerCode(pauli, "XZZXI IXZZX XIXZZ ZXIXZ") # S is not transversal
    
    count = 0
    for P in uniq_codes(pauli, n):

        desc = str(P)
        P = get_dense(P)
        if P*Xt != Xt*P:
            continue
        if P*Zt != Zt*P:
            continue
        if P*St != St*P:
            continue
        if P*Xt == P or P*Xt == -P:
            continue
        if P*Zt == P or P*Zt == -P:
            continue
        if P*Yt == P or P*Yt == -P:
            continue
        print(desc)
        count += 1
    print("count:", count)

    if 0:
        for ops in cross([["I", "S", "X", "Z", "Y"]]*n):
            s = "".join(ops)
            ops = [ns[op] for op in ops]
            op = reduce(matmul, ops)
            if P*op == op*P:
                print(s)


if __name__ == "__main__":

    main()



