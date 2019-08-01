#!/usr/bin/env python3
"""
copied from bruhat.codes
"""

import numpy
from qupy.tool import choose
from qupy.ldpc.solve import array2, find_kernel, dot2, shortstr, span, linear_independent, rank

class Code(object):
    """
        binary linear code, as defined by a generator matrix.
    """
    def __init__(self, G, H=None, d=None, desc="", check=True):
        assert len(G.shape)==2
        self.G = G.copy()
        self.k, self.n = G.shape
        self.d = d
        self.desc = desc

        if H is None:
            H = list(find_kernel(G))
            H = array2(H)
        if H.shape == (0,):
            H.shape = (0, self.n)
        self.H = H.copy()

        if check:
            self.check()

    def check(self):
        G, H = self.G, self.H
        assert rank(G) == len(G)
        A = dot2(H, G.transpose())
        assert A.sum()==0

    def __str__(self):
        desc = ', "%s"'%self.desc if self.desc else ""
        return "Code([[%s, %s, %s]]%s)" % (self.n, self.k, self.d, desc)

    def dump(self):
        G, H = self.G, self.H
        print("G =")
        print(shortstr(G))

        print("H =")
        print(shortstr(H))

    def is_selfdual(self):
        #G = self.G
        #x = dot2(G, G.transpose())
        #return x.sum()==0
        return self.eq(self.get_dual())

    def get_dual(self):
        return Code(self.H, self.G)

    def eq(self, other):
        "Two codes are equal if their generating matrices have the same span."
        G1, G2 = self.G, other.G
        if len(G1) != len(G2):
            return False
        A = dot2(self.H, other.G.transpose())
        B = dot2(other.H, self.G.transpose())
        assert (A.sum()==0) == (B.sum()==0)
        return A.sum() == 0

    def get_distance(self):
        G = self.G
        d = None
        for v in span(G):
            w = v.sum()
            if w==0:
                continue
            if d is None or w<d:
                d = w
        if self.d is None:
            self.d = d
        return d

    def puncture(self, i):
        assert 0<=i<self.n
        G = self.G
        A = G[:, :i]
        B = G[:, i+1:]
        G = numpy.concatenate((A, B), axis=1)
        G = linear_independent(G, check=True)
        return Code(G)



def build(r, m, puncture=False):
    "Build Reed-Muller code"

    assert 0<=r<=m, "r=%s, m=%d"%(r, m)

    n = 2**m # length

    one = array2([1]*n)
    basis = [one]

    vs = [[] for i in range(m)]
    for i in range(2**m):
        for j in range(m):
            vs[j].append(i%2)
            i >>= 1
        assert i==0 

    vs = [array2(v) for v in vs]

    for k in range(r):
        for items in choose(vs, k+1):
            v = one
            #print(items)
            for u in items:
                v = v*u
            basis.append(v)
      
    G = numpy.array(basis)

    code = Code(G, d=2**(m-r), desc="reed_muller(%d, %d)"%(r, m))

    if puncture:
        code = code.puncture(0)

    return code 


def main():
    code = build(1, 4)
    print(code)

if __name__ == "__main__":
    main()


