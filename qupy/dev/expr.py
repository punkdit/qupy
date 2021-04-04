#!/usr/bin/env python3

from qupy.argv import argv


class Expr(object):
    def __init__(self, n=1):
        self.n = n

    def __str__(self):
        return "%s(%s)"%(self.__class__.__name__, self.n)
    __repr__ = __str__

    def __add__(self, other):
        assert self.n == other.n
        return AddExpr(self, other)

    def __sub__(self, other):
        assert self.n == other.n
        return AddExpr(self, RMulExpr(other, -1.))

    def __mul__(self, other):
        assert self.n == other.n
        return MulExpr(self, other)

    def __rmul__(self, other):
        return RMulExpr(self, other)

    def __neg__(self):
        return RMulExpr(self, -1.)


class OpExpr(Expr):
    def __init__(self, name):
        n = len(name)
        Expr.__init__(self, n)
        self.name = name

    def __str__(self):
        return self.name




class AddExpr(Expr):
    def __init__(self, *_items):
        items = []
        for item in _items:
            if isinstance(item, AddExpr):
                items += item.items
            else:
                items.append(item)
        Expr.__init__(self, items[0].n)
        self.items = items

    def __str__(self):
        s = "+".join(str(item) for item in self.items)
        s = s.replace("+-", "-")
        return s
    __repr__ = __str__



class MulExpr(Expr):
    def __init__(self, *_items):
        items = []
        for item in _items:
            if isinstance(item, MulExpr):
                items += item.items
            else:
                items.append(item)
        Expr.__init__(self, items[0].n)
        self.items = items

    def __str__(self):
        return "*".join("(%s)"%(item,) for item in self.items)
    __repr__ = __str__



class RMulExpr(Expr):
    def __init__(self, rhs, val):
        Expr.__init__(self, rhs.n)
        self.rhs = rhs
        self.val = val

    def __str__(self):
        val = self.val
        rhs = self.rhs
        if val == 1:
            return str(rhs)
        if val == -1:
            return "-%s"%(rhs,)
        return "%s*(%s)"%(self.val, self.rhs)
    __repr__ = __str__



def test():

    I = OpExpr("I")
    X = OpExpr("X")
    Z = OpExpr("Z")
    Y = OpExpr("Y")

    print((I+X) * (I-X))


class Space(object):
    def __init__(self, n):
        self.n = n
        self.I = OpExpr("I"*n)

    def make_zop(self, idxs):
        n = self.n
        name = ['I'] * n
        for idx in idxs:
            name[idx] = 'Z'
        name = ''.join(name)
        return OpExpr(name)

    def make_xop(self, idxs):
        n = self.n
        name = ['I'] * n
        for idx in idxs:
            name[idx] = 'X'
        name = ''.join(name)
        return OpExpr(name)


from qupy.ldpc.solve import (
        identity2, kron, dot2, rank, int_scalar,
        parse, remove_dependent, zeros2, rand2, shortstr, all_codes,
        find_kernel )


class Code(object):
    def __init__(self, Hz, Hx, Lz=None, Lx=None, **kw):
        self.__dict__.update(kw)
        mz, n = Hz.shape
        mx, nx = Hx.shape
        assert n==nx
        assert rank(Hz)==mz
        assert rank(Hx)==mx
        xstabs = []
        zstabs = []
        space = Space(n)

        for i in range(mz):
            idxs = [j for j in range(n) if Hz[i, j]]
            op = space.make_zop(idxs)
            zstabs.append(op)

        for i in range(mx):
            idxs = [j for j in range(n) if Hx[i, j]]
            op = space.make_xop(idxs)
            xstabs.append(op)

        # code projector:
        P = None
        for op in zstabs + xstabs:
            A = (space.I + op)
            P = A if P is None else A*P
        print(P)

    def check(self):
        pass




def main_8T():
    """
    _Transversal T gate on the [[8,3,2]] colour code.
    https://earltcampbell.com/2016/09/26/the-smallest-interesting-colour-code
    https://arxiv.org/abs/1706.02717
    """

    Hz = parse("1111.... 11..11.. 1.1.1.1. 11111111")
    Hx = parse("11111111")
    Lz = parse("1...1... 1.1..... 11......")
    Lx = parse("1111.... 11..11.. 1.1.1.1.")

    code = Code(Hz, Hx, Lz, Lx)
    code.check()



if __name__ == "__main__":

    _seed = argv.get("seed")
    if _seed is not None:
        seed(_seed)
        numpy.random.seed(_seed)

    name = argv.next() or "test"
    fn = eval(name)
    fn()

    print("%s(): OK"%name)


