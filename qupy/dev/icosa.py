#!/usr/bin/env python3

"""
build commutative subgroup of tensor power of
binary platonic groups.
These should not contain -I, so they can
serve as a stabilizer group.
"""

from random import randint, choice, seed
from functools import reduce
from operator import matmul, mul, add

#from scipy.spatial import KDTree, cKDTree
import numpy

from qupy.dense import Qu
from qupy.util import mulclose
from qupy.tool import cross, write
from qupy.argv import argv
#from qupy.dev.su2 import Icosa
from qupy.dev import groups


EPSILON=1e-8


class Group(object):
    def __init__(self, dense):
        n = len(dense)
        mul = {}
        for i, g in enumerate(dense):
          for j, h in enumerate(dense):
            gh = g*h
            k = dense.index(gh)
            mul[i, j] = k
    
        ident = None
        for i in range(n):
            for j in range(n):
                if mul[i, j] != j:
                    break
            else:
                assert ident is None
                ident = i
    
        negi = None
        for i in range(n):
            for j in range(n):
                if mul[i, j] != mul[j, i]:
                    break
            else:
                #print("central:", i)
                if i==ident:
                    pass
                else:
                    assert negi is None
                    negi = i
    
        canonical = {ident:ident, negi:ident}
        for i in range(n):
            if i in canonical:
                continue
            j = mul[i, negi]
            assert j not in canonical
            canonical[i] = i
            canonical[j] = i
    
        assert len(canonical)==n
        assert len(set(canonical.values()))==n//2

        self.n = n 
        self.mul = mul
        self.ident = ident
        self.negi = negi
        self.canonical = canonical

        self.dense = dense

        ops = []
        non_central = []
        for i in range(n):
            idx = canonical[i]
            phase = +1 if idx==i else -1
            op = Tensor(self, [idx], phase)
            ops.append(op)
            if idx != ident:
                non_central.append(op)
        self.ops = ops
        self.non_central = non_central

    def get_ident(self, n=1):
        return Tensor(self, [self.ident]*n, +1)

    def __getitem__(self, idx):
        return self.ops[idx]

    def __len__(self):
        return len(self.ops)

    def get_dense(self, op):
        dense = self.dense
        ops = [dense[i] for i in op.idxs]
        assert len(ops)<=10, len(ops)
        op = op.phase * reduce(matmul, ops)
        return op


class Tensor(object):
    def __init__(self, G, idxs, phase):
        self.G = G
        self.idxs = idxs
        self.phase = phase
        self.check()

    def check(self):
        G = self.G
        idxs = self.idxs
        for idx in idxs:
            assert G.canonical[idx] == idx

    def __mul__(self, other):
        assert self.G is other.G
        assert len(self.idxs)==len(other.idxs)
        G = self.G
        n = len(self.idxs)
        idxs = []
        phase = self.phase * other.phase
        for i in range(n):
            idx = G.mul[self.idxs[i], other.idxs[i]]
            jdx = G.canonical[idx]
            if jdx!=idx:
                phase *= -1
            idxs.append(jdx)
        return Tensor(G, idxs, phase)

    def __rmul__(self, phase):
        assert phase==1 or phase==-1
        return Tensor(self.G, list(self.idxs), phase*self.phase)

    def __neg__(self):
        return -1*self

    def __matmul__(self, other):
        assert self.G is other.G
        return Tensor(self.G, self.idxs+other.idxs, self.phase*other.phase)

    def __eq__(self, other):
        return self.phase==other.phase and self.idxs==other.idxs

    def __ne__(self, other):
        return self.phase!=other.phase or self.idxs!=other.idxs

    def __hash__(self):
        return hash((self.phase, self.idxs))

    def __repr__(self):
        return "Tensor(%s, %s)"%(self.idxs, self.phase)

    def __str__(self):
        ops = []
        for idx in self.idxs:
            if idx==self.G.ident:
                ops.append("__")
            else:
                ops.append(str(idx).rjust(2))
        s = "[%s]"%(",".join(ops))
        if self.phase != 1:
            s = "-"+s
        return s


def show_spec(P):
    print("spec:", end=" ")
    for val, vec in P.eigs():
        if abs(val) > EPSILON:
            if abs(val.imag) < EPSILON:
                val = val.real
            if abs(val - round(val)) < EPSILON:
                val = int(round(val))
            print(val, end=" ")
    print()

def find_errors(G, P, degree):
    I = G.get_ident(1)
    nI = -I
    for i in range(degree):
        for g in G:
            if g==I or g==nI:
                continue
            ops = [I] * degree
            ops[i] = g
            op = reduce(matmul, ops)
            if op.phase == -1:
                continue
            #if op in S:
            #    write('.')
            U = G.get_dense(op)
            if U*P == P*U and P*U != P:
                #write('*')
                print(op)
    print()


def main():

    name = argv.get("G", "icosa")
    G = groups.build(name)

    def get_reflects(G, I):
        reflects = []
        count = 0
        for ops in cross((G,)*degree):
            op = reduce(matmul, ops)
            if op*op == I:
                count += 1
                reflects.append(op)
        reflects.remove(I)
        reflects.remove(nI)
        return reflects

    def rand_reflect(G, I, degree, weight=None):
        if weight is None:
            weight = degree
        while 1:
            #ops = [choice(G) for i in range(degree)]
            idxs = list(range(degree))
            ops = [G.get_ident() for i in idxs]
            for j in range(weight):
                idx = choice(idxs)
                idxs.remove(idx)
                ops[idx] = choice(G.non_central)
            op = reduce(matmul, ops)
            if op*op == I and op != I and op.phase == 1:
                return op

    def mk_stab(reflects):
        # build commutative subgroup from these
        size = 0
        for trials in range(100):
            remain = list(reflects)
            gen = []
            S = []
            while remain:
                idx = randint(0, len(remain)-1)
                a = remain.pop(idx)
                for b in gen:
                    if a*b != b*a:
                        break
                else:
                    if a not in S:
                        S1 = mulclose(gen + [a])
                        if nI not in S1:
                            S = S1
                            gen.append(a)
            
            if len(S) > size:
                #print("|S| =", len(S))
                size = len(S)
    
        for op in S:
            assert op*op == I
        return gen, S

    print("|G| =", len(G))
    print("building...", end="", flush=True)
    G = Group(G)
    print("done")

    degree = argv.get("degree", 2)
    weight = argv.get("weight", degree)


    def build_spin_chain(G):
        I = G.get_ident()
        In = G.get_ident(degree)
        nt_ops = [op for op in G if op!=I and op.phase==1]
        while 1:
            while 1:
                A = [I]*degree
                B = [I]*degree
        
                for i in range(weight):
                    a = choice(nt_ops)
                    A[i] = a
                    B[(i+1)%degree] = a
                A = reduce(matmul, A)
                B = reduce(matmul, B)
                if A*B == B*A:
                    break
        
            gen = [A, B]
            for i in range(2, degree):
                idxs = [A.idxs[(-i+j)%degree] for j in range(degree)]
                op = Tensor(G, idxs, A.phase)
                gen.append(op)
            print("gen:")
            for op in gen:
                print("  ", op)
        
            S = mulclose(gen)
            print("|S| =", len(S))
    
            #ops = [G.get_dense(op) for op in S]
            #P = reduce(add, ops)
            P = G.get_dense(S[0])
            for op in S[1:]:
                P = P + G.get_dense(op)
            r = P.norm()
            if abs(r) > EPSILON:
                break
            write(".")
    
        show_spec(P)
        find_errors(G, P, degree)
        return

    I = G.get_ident(degree)
    nI = -I

    n_reflects = argv.get("reflects", 4)

    reflects = []
    while 1:
        reflects += [rand_reflect(G, I, degree, weight) for i in range(n_reflects)]
    
        if 0:
            #print("reflects:")
            idxs = set()
            for op in reflects:
                for idx in op.idxs:
                    idxs.add(idx)
            idxs = list(idxs)
            idxs.sort()
            ops = [G[idx] for idx in idxs]
            H = mulclose(ops)
            print("reflects:", idxs)
            print("|H|:", len(H))
    
        gen, S = mk_stab(reflects)
        write("%s,"%len(S))
        if len(S) >= 2**degree:
            break
    print()
    print("gen:")
    for op in gen:
        print("  ", op)
    
    gen.pop(0)
    S = mulclose(gen)

    # build projector onto codespace
    ops = [G.get_dense(op) for op in S]
    P = reduce(add, ops)
    show_spec(P)
        

    find_errors(G, P, degree)




if __name__ == "__main__":

    _seed = argv.seed
    if _seed is not None:
        seed(_seed)
        numpy.random.seed(_seed)

    main()


