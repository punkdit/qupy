#!/usr/bin/env python3

from random import randint, choice
from functools import reduce
from operator import matmul

from scipy.spatial import KDTree, cKDTree
import numpy

from qupy.dense import Qu
from qupy.util import mulclose, show_spec
from qupy.tool import cross
from qupy.argv import argv
#from qupy.dev.su2 import Icosa

EPSILON=1e-8



def cyclotomic(n):
    return numpy.exp(2*numpy.pi*1.j/n)


def build_tetra():
    " binary _tetrahedral group "

    i = cyclotomic(4)

    gen = [
        [[(-1+i)/2, (-1+i)/2], [(1+i)/2, (-1-i)/2]],
        [[-1/2-1/2*i, -1/2-1/2*i], [1/2-1/2*i, -1/2+1/2*i]]
    ]    

    gen = [Qu((2, 2), 'ud', v) for v in gen] 

    Tetra = mulclose(gen)
    assert len(Tetra)==24

    return Tetra


def build_octa():
    " binary _octahedral group "

    x = cyclotomic(8)

    a = x-x**3 # sqrt(2)
    i = x**2 

    gen = [
        [[(-1+i)/2,(-1+i)/2], [(1+i)/2,(-1-i)/2]], 
        [[(1+i)/a,0], [0,(1-i)/a]]
    ]    
    gen = [Qu((2, 2), 'ud', v) for v in gen] 
    octa_gen = gen

    Octa = mulclose(gen)
    assert len(Octa)==48
    #print("Octa:", Octa.words.values())

    return Octa


def build_icosa():
    " binary _icosahedral group "

    v = cyclotomic(10)
    z5 = v**2 
    a = 2*z5**3 + 2*z5**2 + 1 #sqrt(5)
    gen = [
        [[z5**3, 0], [0, z5**2]], 
        [[(z5**4-z5)/a, (z5**2-z5**3)/a], 
        [(z5**2-z5**3)/a, -(z5**4-z5)/a]]
    ]    
    
    gen = [Qu((2, 2), 'ud', v) for v in gen] 

    X = Qu((2, 2), 'ud', [[0., 1], [1., 0.]])
    Z = Qu((2, 2), 'ud', [[1., 0], [0., -1.]])

    G = mulclose(gen)

    assert len(G)==120

    assert X*Z in G
    assert X not in G
    assert Z not in G

    return G


def build_pauli():

    X = Qu((2, 2), 'ud', [[0., 1], [1., 0.]])
    Z = Qu((2, 2), 'ud', [[1., 0], [0., -1.]])
    gen = [X, Z]

    G = mulclose(gen)

    assert len(G)==8

    return G


class Group(object):
    def __init__(self, items):
        n = len(items)
        mul = {}
        for i, g in enumerate(items):
          for j, h in enumerate(items):
            gh = g*h
            k = items.index(gh)
            mul[i, j] = k
        #print(mul)
    
        ident = None
        for i in range(n):
            for j in range(n):
                if mul[i, j] != j:
                    break
            else:
                assert ident is None
                ident = i
        #print("ident:", ident)
    
        negi = None
        for i in range(n):
            #central = True
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
        #print("negi:", negi)
    
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

        items = []
        for i in range(n):
            idx = canonical[i]
            phase = +1 if idx==i else -1
            op = Tensor(self, [idx], phase)
            items.append(op)
        self.items = items

    def get_ident(self, n):
        return Tensor(self, [self.ident]*n, +1)

    def __getitem__(self, idx):
        return self.items[idx]



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

    def __str__(self):
        return "Tensor(%s, %s)"%(self.idxs, self.phase)



def main():

    name = argv.get("G", "icosa")

    if name == "icosa":
        G = build_icosa()
    elif name == "tetra":
        G = build_tetra()
    elif name == "octa":
        G = build_octa()
    elif name == "pauli":
        G = build_pauli()
    else:
        return

    I = Qu((2, 2), 'ud', [[1., 0], [0., 1.]])

    def get_order(g, I):
        order = 1
        h = g
        while h != I:
            h = g*h
            order += 1
        return order

    #for g in G:
    #    print(get_order(g), end=" ")
    #print()


    def get_intops(G, I, verbose=False):
        #ops = []
        for g in G:
            vals = [v for v,_ in g.eigs()]
            for v in vals:
                if abs(v.imag)>EPSILON or abs(abs(v.real)-1) > EPSILON:
                    break
            else:
                #ops.append(g)
                if verbose:
                    print([int(round(v.real)) for v in vals])
                    #print(vals)
                    #print([abs(v.imag)<EPSILON for v in vals])
                if g*g != I:
                    print(g.v)
                    print((g*g).v)
                    show_spec(g)
                    assert 0
                yield g
        #return ops

    #def get_reflects(G, I):
    #    ops = [g for g in G if g*g==I]
    #    return ops

    def get_reflects(G, I):
        for g in G:
            if g*g==I:
                yield g

    def count(items):
        i = 0
        for _ in items:
            i += 1
        return i


    if argv.get_reflects:
        ops = get_intops(G, I)
        print(count(ops))
        #return
    
        I2 = I@I
    
        G2 = [g@h for g in G for h in G]
        #print("G2:", len(G2))
        #ops = get_intops(G2, I2)
        #print(count(ops))
    
        ops = get_reflects(G2, I2)
        print(count(ops))
        #print(len(mulclose(ops)))
    
        #G3 = [g@h for g in G2 for h in G]
        ops = get_reflects((g@h for g in G2 for h in G), I2@I)
        print(count(ops))
        #print(len(mulclose(ops)))
    
        ops = get_reflects((g@h for g in G2 for h in G2), I2@I2)
        print(count(ops))

    if argv.find_uniq:
        G2 = []
        for g in G:
          for h in G:
            gh = g@h
            v = gh.v
            v = v.flatten()
            G2.append(v)
        tree = KDTree(G2, leafsize=10)
        N = len(G2)
        remain = set(range(N))
        print(N)
        uniq = []
        while remain:
            idx = iter(remain).__next__()
            remain.remove(idx)
            uniq.append(idx)
            v = G2[idx]
            k = 3
            dists, idxs = tree.query(v, k, EPSILON)
            assert idxs[0] == idx
            for i in range(1, k):
                if dists[i] < EPSILON:
                    remain.remove(idxs[i])
        print(len(uniq))
    
    if 0:
        g = G[0]
        g0 = G[1]@G[2]@G[3]
        G2 = []
        count = 0
        for g in G:
          for h in G:
            gh = g@h
            for k in G:
                ghk = gh@k
                #v = ghk.v.flatten()
                #G2.append(v)
                if ghk == g0:
                    print(".", end=" ", flush=True)
                    count += 1
        print("count:", count)

    G = Group(G)

    degree = argv.get("degree", 2)

    I = G.get_ident(degree)
    nI = -I
    reflects = []
    count = 0
    for ops in cross((G,)*degree):
        op = reduce(matmul, ops)
        if op*op == I:
            count += 1
            reflects.append(op)
    print("count:", count)

    reflects.remove(I)
    reflects.remove(nI)

    size = 0
    #while 1:
    for trials in range(1000):
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
            print("|S| =", len(S))
            size = len(S)
    



if __name__ == "__main__":

    main()


