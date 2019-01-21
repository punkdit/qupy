#!/usr/bin/env python3

import sys
import math

from operator import mul, matmul

import numpy
import numpy.random
from functools import reduce

try:
    import scipy.linalg
except ImportError:
    pass


from qupy.scalar import scalar, EPSILON, MAX_TENSOR

zero = scalar(0.)

from qupy import abstract
from qupy.abstract import AbstractQu, Space, genidx


def find(items, item, count):
    #print "FIND(%s, %s, %s)"%(items, item, count)
    idx = 0
    while idx < len(items):
        if items[idx] == item:
            if count == 0:
                #print "\tFIND", idx
                return idx
            count -= 1
        idx += 1
    raise ValueError("find(%s, %s, %s)"%(items, item, count))


def shortstr(x):
    if abs(x.imag) < EPSILON:
        x = x.real
        if x == int(x):
            x = int(x)
        xs = str(x)
    elif abs(x.real) < EPSILON:
        x = x.imag
        if x == int(x):
            x = int(x)
        xs = '%sj'%str(x)
    else:
        xs = str(x)
    return xs


class MonoidalFlatten(object):
    """ Squash high rank tensors down into just rank-2 tensor
        (of higher dimension). We also need to reverse this.
        We preserve valence.
    """
    def __init__(self, space):
        self.space = space
        self.rank = rank = len(space.shape)
        up_idxs = [idx for idx in range(rank) if space.valence[idx] == 'u']
        dn_idxs = [idx for idx in range(rank) if space.valence[idx] == 'd']
        self.perm = perm = up_idxs + dn_idxs
        self.up_shape = [space.shape[idx] for idx in up_idxs]
        self.dn_shape = [space.shape[idx] for idx in dn_idxs]
        if self.up_shape and self.dn_shape:
            shape = (reduce(mul, self.up_shape), reduce(mul, self.dn_shape))
            valence = 'ud'
        elif self.up_shape:
            shape = (reduce(mul, self.up_shape),)
            valence = 'u'
        elif self.dn_shape:
            shape = (reduce(mul, self.dn_shape),)
            valence = 'd'
        else:
            shape = ()
            valence = ''

#        shape = (reduce(mul, self.up_shape, 1), reduce(mul, self.dn_shape, 1))
#        if shape[0] == 1:
#            valence = 'd'
#            shape = shape[1],
#        elif shape[1] == 1:
#            valence = 'u'
#            shape = shape[0],
#        else:
#            valence = 'ud'
        self.shape = shape
        self.valence = valence
        self.target_space = Space(self.shape, self.valence)

    def __mul__(self, other):
        space = self.space * other.space
        return MonoidalFlatten(space)

    def do(self, box):
        box = box.clone()
        box.permute(self.perm)
        #print "MonoidalFlatten:", self.perm, "-->", box.space
        v = box.v.flatten()
        v.shape = self.shape
        box = Qu(v.shape, self.valence, v)
        #print "MonoidalFlatten:", "-->", box.space
        if box.rank == 2:
            box = Gate.promote(box)
        return box

    def undo(self, box):
        assert box.shape == self.shape, "%s != %s"%(box.shape, self.shape)
        assert box.valence == self.valence
        v = box.v.copy()
        v.shape = tuple(self.up_shape+self.dn_shape)
        box = Qu(v.shape, 'u'*len(self.up_shape)+'d'*len(self.dn_shape),
            v, nocopy=True)
        iperm = [self.perm.index(i) for i in range(self.rank)]
        box.permute(iperm)
        assert box.space == self.space, "%s != %s" % (box.space, self.space)
        return box


class Qu(AbstractQu):

    def __init__(self, shape, valence, v=None, nocopy=False, dtype=None):
        """
        """
        if type(shape) is int:
            shape = (shape,)
        assert type(valence) is str
        AbstractQu.__init__(self, shape, valence)
        assert len(shape) <= MAX_TENSOR, "ouch, this is getting big: rank=%d"%len(shape)
        if dtype is None:
            dtype = scalar
        if nocopy:
            assert v is not None
            assert isinstance(v, numpy.ndarray)
            self.v = v
        else:
            assert reduce(mul, shape, 1) < 2**MAX_TENSOR, "too big: %d"%reduce(mul, shape, 1)
            self.v = numpy.zeros(shape, dtype)
            if v is not None:
                self.v[()] = v

    @classmethod
    def bits(cls, bits, arity=2):
        rank = len(bits)
        w = cls((arity,)*rank, 'u'*rank)
        key = []
        for idx, bit in enumerate(bits):
            if bit == '0':
                key.append(0)
            elif bit == '1':
                key.append(1)
            else:
                raise ValueError("wah! %s"%bits)
        assert len(key) == len(w.shape)
        #print "BITS:", w
        #print "KEY:", key
        w[tuple(key)] = 1.
        #print "BITS:", w
        return w

    @classmethod
    def basis(cls, shape, valence):
        elems = []
        for idx in genidx(shape):
            v = cls(shape, valence)
            v[idx] = 1.
            elems.append(v)
        return elems

    def is_square(self):
        ups = [i for i, ud in enumerate(self.valence) if ud=='u']
        dns = [i for i, ud in enumerate(self.valence) if ud=='d']
        return [self.shape[i] for i in ups] == [self.shape[i] for i in dns]

    def eigs(H, verbose=False, hermitian=False):
        if not H.is_square() or not H.rank:
            raise ValueError(H.shape)

        flatop = H.get_flatop()
        H1 = flatop.do(H)

        if verbose:
            print("eig...", end=' ', flush=True)
        if hermitian:
            eigvals, eigvecs = numpy.linalg.eigh(H1.v)
        else:
            eigvals, eigvecs = numpy.linalg.eig(H1.v)
        N = len(eigvals)
        if verbose:
            print("done")

        idxs = [i for i, ud in enumerate(H.valence) if ud=='d']
        shape = [H.shape[i] for i in idxs]

        items = []
        for i in range(N):
            vec = eigvecs[:, i]
            vec.shape = shape
            val = eigvals[i]
            v = Qu(shape, 'u'*len(shape), vec, True)
            items.append((val, v))
        return items

    @property
    def shape(self):
        assert hasattr(self.v, 'shape'), self.v
        assert self.v.shape == self.space.shape, \
            "%r != %r"%(self.v.shape, self.space.shape)
        assert self.v.shape == self.space.shape, "v changed shape but not my space!"
        return self.v.shape

    def clone(self):
        w = Qu(self.shape, self.valence, self.v)
        w.__class__ = self.__class__ # Woah..
        return w

    def shortstr(self):
        ss = []
        for idx in genidx(self.shape):
            if abs(self[idx]) > EPSILON:
                ss.append("%s : %s" % (idx, shortstr(self[idx])))
        return "{%s}"%', '.join(ss)

    @property
    def nnz(self):
        n = 0
        for idx in genidx(self.shape):
            if self[idx] != 0.+0.j:
                n += 1
        return n

    def nzidxs(self):
        idxs = []
        for idx in genidx(self.shape):
            if self[idx] != 0.+0.j:
                idxs.append(idx)
        return idxs

    def tosparse(self):
        from sparse import SparseQu
        box = SparseQu(self.shape, self.valence)
        for idx in genidx(self.shape):
            r = self.v[idx] 
            if r != 0.+0.j:
                box.data[idx] = r
        return box

    def todense(self):
        return self

    def flattuple(self):
        values = []
        for idx in genidx(self.shape):
            values.append(self.v[idx])
        return tuple(values)

    def __hash__(self):
        return hash((self.space, self.flattuple()))

    def flatstr(self):
        ss = []
        for idx in genidx(self.shape):
            if self[idx] != 0.+0.j:
                s_idx = ''.join('%d'%i for i in idx)
                #s_idxs = []
                #for i in range(self.rank):
                #    if self.valence[i] == 'u':
                #        s_idxs.append('|%d>'%idx[i])
                #    else:
                #        s_idxs.append('<%d|'%idx[i])
                #s_idx = ''.join(s_idxs)
                ss.append("%s : %s\n" % (s_idx, self[idx]))
        return ''.join(ss)

    def sqstr(A, space=0):
        op = A.get_flatop()
        B = op.do(A)
        m, n = B.shape
        ss = [[None for i in range(m)] for j in range(n)]
        for i in range(m):
            for j in range(n):
                x = B[i, j]
                xs = shortstr(x)
                ss[i][j] = xs
                space = max(len(xs), space)
        ss = [' '.join('%*.*s' % (space, space, entry) for entry in row)
            for row in ss]
        s = '\n'.join(ss)
        return s

    def reshape(self, shape, valence, nocopy=False):
        v = self.v.view()
        v.shape = shape
        return Qu(shape, valence, v, nocopy)

    def swap(self, *pairs):
        perm = list(range(self.rank))
        valence = list(self.valence)
        for i, j in pairs:
            perm[i], perm[j] = perm[j], perm[i]
            assert valence[j] == valence[i]
            #valence[i], valence[j] = valence[j], valence[i]
        valence = ''.join(valence)
        v = self.v.transpose(perm)
        box = Qu(v.shape, valence, v)
        return box

    def permute(self, idxs):
        # XXX i don't much like mutability here...
        self.valence = ''.join(self.valence[i] for i in idxs)
        self.v = self.v.transpose(idxs)
        self.space.shape = self.v.shape # wah!

    def permuted(self, idxs):
        valence = ''.join(self.valence[i] for i in idxs)
        v = self.v.transpose(idxs)
        A = Qu(v.shape, valence, v, nocopy=True)
        return A

    def swap2(self, i, j):
        """ swap i'th input with j'th input
            and i'th output with j'th output
        """
        if i == j:
            return
        if i > j:
            i, j = j, i
        #print "SHIFT:", self.valence, i, j
        assert self.valence.count('u') > i
        assert self.valence.count('u') > j
        assert self.valence.count('d') > i
        assert self.valence.count('d') > j
        idxs = list(range(self.rank))
        for updn in 'ud':
            idx, jdx = find(self.valence, updn, i), find(self.valence, updn, j)
            #idxs[idx], idxs[jdx] = idxs[jdx], idxs[idx]
            assert idx < jdx
            assert self.valence[idx] == self.valence[jdx] == updn
            #print "SHIFT:", updn, idx, jdx
            #idxs.insert(jdx-1, idxs.pop(idx))
            idxs[idx], idxs[jdx] = idxs[jdx], idxs[idx]
        #print "SHIFT:", idxs
        self.permute(idxs)

    def adj(self, pair=(0, 1)):
        i, j = pair
        assert i != j
        valence = self.valence
        assert valence[i] != valence[j]
        perm = list(range(self.rank))
        perm[i], perm[j] = perm[j], perm[i]
        v = self.v.transpose(perm)
        v = v.conj()
        box = Qu(v.shape, valence, v)
        return box

    def dag(A):
        space = ~A.space
        v = A.v.conj()
        box = Qu(space.shape, space.valence, v, nocopy=True)
        return box
    __invert__ = dag

    def __str__(self):
        s = "%s(%s, %s, %s)"%(self.__class__.__name__, self.shape, self.valence, self.v)
        s = s.replace('\n', ' ')
        return s
    __repr__ = __str__

    def is_close(self, other, epsilon=EPSILON):
        v0, v1 = self.v, other.v
        perm = self.space.unify(other.space)
        if perm is None:
            raise ValueError("space mismatch: %s and %s" % (self.space, other.space))
        v2 = v0.transpose(perm)
        a = numpy.abs(v1-v2)
        r = numpy.sum(a)
        return r < epsilon
    __eq__ = is_close # good idea?

    def norm(self):
        v = self.v
        v = v.ravel()
        v = v * v.conj()
        v = v.real
        r = v.sum()
        r = numpy.sqrt(r)
        return r

    def normalize(self):
        r = self.norm()
        self /= r

    def normalized(self):
        r = self.norm()
        return self / r

    def __getitem__(self, idx):
        item = self.v[idx]
        if type(item) is numpy.ndarray:
            # XXX
            assert 0, "not implemented.. idx=%r, item.shape=%r"%(idx, item.shape)
        return item

    def __setitem__(self, idx, value):
        if isinstance(value, Qu):
            value = value.v
        self.v[idx] = value

    def __add__(A, B):
        C = A.unify(B)
        if C is None:
            raise ValueError("space mismatch: %s <-> %s"%(A.space, B.space))
        C = C.clone()
        C.v += B.v
        return C

    def __iadd__(A, B):
        C = A.unify(B)
        if C is None:
            raise ValueError("space mismatch: %s <-> %s"%(A.space, B.space))
        C.v += B.v
        return A

    def __sub__(A, B):
        C = A.unify(B)
        if C is None:
            raise ValueError("space mismatch: %s <-> %s"%(A.space, B.space))
        C = C.clone()
        C.v -= B.v
        return C

    def __isub__(A, B):
        C = A.unify(B)
        if C is None:
            raise ValueError("space mismatch: %s <-> %s"%(A.space, B.space))
        C.v -= B.v
        return A

    def __neg__(A):
        A = A.clone()
        A.v *= -1
        return A

    def __rmul__(self, r):
        #print("\n__rmul__", r)
        w = self.clone()
        w.v *= r
        #print("__rmul__: done")
        return w

    def __imul__(A, r):
        A.v *= r
        return A

    def __div__(A, r):
        return (1./r)*A
    __floordiv__ = __div__
    __truediv__ = __div__

    def __idiv__(A, r):
        A.v /= r
        return A
    __ifloordiv__ = __idiv__
    __itruediv__ = __idiv__

    def tensor(A, B):
        if not isinstance(B, Qu):
            return B*A
        v = numpy.outer(A.v, B.v)
        v.shape = A.shape + B.shape
        w = Qu(A.shape + B.shape, A.valence + B.valence, v, True)
        #w.v[:] = v
        return w
    #__mul__ = tensor
    __matmul__ = tensor

    def contract(self, i, j):
        assert 0 <= i < self.rank
        assert 0 <= j < self.rank
        assert i != j
        assert self.shape[i] == self.shape[j], repr((self.shape, i, j))
        assert self.valence[i] != self.valence[j]
        v = self.v
        v = numpy.trace(self.v, 0, i, j)
        valence = list(self.valence)
        valence.pop(max(i, j))
        valence.pop(min(i, j))
        box = Qu(v.shape, ''.join(valence), v, True)
        return box

    def apply(self, other):
        assert self.valence[-1] != other.valence[0]
        assert self.shape[-1] == other.shape[0]
        n = self.shape[-1]
        box = Qu(self.shape[:-1] + other.shape[1:], self.valence[:-1] + other.valence[1:])
        for idx0 in genidx(self.shape[:-1]):
            for idx1 in genidx(other.shape[1:]):
                r = 0.+0.j
                for i in range(n):
                    r += self[idx0+(i,)] * other[(i,)+idx1]
                box[idx0+idx1] = r
        return box

    def __mul__(A, B):
        dn_count = A.valence.count('d')
        up_count = B.valence.count('u')
        if dn_count != up_count:
            raise ValueError("cannot match valence %s * %s" % (A.valence, B.valence))

        idxs = [i for i in range(A.rank) if A.valence[i]=='d']
        jdxs = [i for i in range(B.rank) if B.valence[i]=='u']

        if [A.shape[i] for i in idxs] != [B.shape[i] for i in jdxs]:
            raise ValueError("cannot match shape")

        if not isinstance(B, Qu):
            return NotImplemented

        v = numpy.tensordot(A.v, B.v, [idxs, jdxs])

        idxs = [i for i in range(A.rank) if A.valence[i]=='u']
        jdxs = [i for i in range(B.rank) if B.valence[i]=='d']

        shape = tuple([A.shape[i] for i in idxs]+[B.shape[i] for i in jdxs])
        valence = 'u'*len(idxs) + 'd'*len(jdxs)

        assert shape==v.shape

        if len(shape)==0:
            C = v[()] # peel out the scalar element
        else:
            C = Qu(shape, valence, v, True)

        return C

#    def __rmul__(self, other):
#        return other * self # __rmul__

    def __pow__(self, count):
        if type(count) != int:
            raise ValueError
        if count == 0:
            return 1.
        if count == 1:
            return self
        A = self
        while count > 1:
            A = self * A
            count -= 1
        return A

    def dot(A, B):
        return (~A*B)

    @classmethod
    def bell_basis(self, n=2):
        raise NotImplementedError
        return basis

    def decode(self):
        "decode as bit string"
        v = self.v
        v = v.ravel()
        idx = v.real.argmax()
        assert is_close(v[idx], 1.), "XXX not implemented"
        rank = self.rank
        bs = ['0' for i in range(rank)]
        for i in range(rank):
            if (idx & (1<<i)):
                bs[rank-i-1] = '1'
        return ''.join(bs)

    @classmethod
    def random(cls, shape, valence):
        v_r = numpy.random.normal(size=shape)
        v_i = numpy.random.normal(size=shape)
        v_i = v_i.astype(scalar)
        v_i *= 1.j
        v = v_r + v_i
        box = Qu(shape, valence, v)
        return box

#    def control(self):
#        "0-th bit is control, 1st bit is self"
#        n = self.rank
#        I = Gate.identity(n)
#        A = Gate.dyads[0]@I + Gate.dyads[1]@self
#        return A

    def control(self, target=1, *source):
        source = source or (0,)
        if target in source:
            raise ValueError("target in source")

        rank = max(target, max(source))+1

        A = Qu((2,)*(2*rank), 'ud'*rank)

        for ctrl in genidx((2,)*len(source)):
            factors = [Gate.I] * rank
            for idx, bit_idx in enumerate(source):
                factors[bit_idx] = Gate.dyads[ctrl[idx]]
            if 0 not in ctrl: # all 1's
                factors[target] = self
            A += reduce(matmul, factors)
        return A

    def get_flatop(self):
        op = MonoidalFlatten(self.space)
        return op

    def flat(A):
        op = A.get_flatop()
        A = op.do(A)
        return A

    @classmethod
    def random_hermitian(cls, space):
        flatop = MonoidalFlatten(space)
        space = flatop.target_space
        A = Gate.random_hermitian(space)
        A = flatop.undo(A)
        return A

    @classmethod
    def random_unitary(cls, space):
        flatop = MonoidalFlatten(space)
        space = flatop.target_space
        A = Gate.random_unitary(space)
        A = flatop.undo(A)
        return A

    @classmethod
    def identity(cls, space):
        flatop = MonoidalFlatten(space)
        space = flatop.target_space
        A = Gate.identity(space)
        A = flatop.undo(A)
        return A

    def is_hermitian(A, epsilon=EPSILON):
        flatop = MonoidalFlatten(A.space)
        A = flatop.do(A)
        return A.is_hermitian(epsilon)

    def is_unitary(A, epsilon=EPSILON):
        flatop = MonoidalFlatten(A.space)
        A = flatop.do(A)
        return A.is_unitary(epsilon)

    def is_identity(A, epsilon=EPSILON):
        n = A.shape[0]
        assert A.shape == (n, n)
        for i in range(n):
            for j in range(n):
                if i==j:
                    if abs(A.v[i, j]-1.)>epsilon:
                        return False
                else:
                    if abs(A.v[i, j])>epsilon:
                        return False
        return True

    def trace(A):
        flatop = MonoidalFlatten(A.space)
        A = flatop.do(A)
        return A.trace()

    def is_pure(A, epsilon=EPSILON):
        flatop = MonoidalFlatten(A.space)
        A = flatop.do(A)
        return A.is_pure(epsilon)

    def expm(A, order=10):
        assert 0, "XXX I dont think this works... XXX"
        flatop = A.get_flatop()
        A = flatop.do(A)
        #A = A.expm(order)
        A = flatop.undo(A)
        return A

    def evolution_operator(H, t, order=10):
        flatop = H.get_flatop()
        H = flatop.do(H)
        U = H.evolution_operator(t, order)
        U = flatop.undo(U)
        return U



abstract.Qu = Qu # XX monkey patch... ugly..

class Vector(Qu):
    def __init__(self, v):
        Qu.__init__(self, (len(v),), 'u')
        self.v[:] = v
Ket = Vector

class CoVector(Qu):
    def __init__(self, v):
        Qu.__init__(self, (len(v),), 'd')
        self.v[:] = v
Bra = CoVector


class Bit(Qu):
    def __init__(self, v=None):
        Qu.__init__(self, (2,), 'u')



off = Bit()
off[0] = 1.

on = Bit()
on[1] = 1.

bits = Qu.bits
up, dn = 'u', 'd'


def bitvec(*bits, base=2):
    n = len(bits)
    v = Qu((base,)*n, 'u'*n)
    v[bits] = 1.
    return v




class Gate(Qu):
    def __init__(self, shape, v=None, nocopy=False):
        if type(shape) is int:
            shape = (shape, shape)
        assert len(shape) == 2
        valence = 'ud'
        Qu.__init__(self, shape, valence, v, nocopy)

    @classmethod
    def promote(cls, box):
        if box.__class__ is cls:
            return box
        assert box.valence == 'ud' or box.valence == 'du', box.valence
        box = cls(box.shape, box.v, nocopy=True)
        return box

    @classmethod
    def random_hermitian(cls, space):
        if type(space) is int:
            n = space
            space = Space((n, n), 'ud')
        assert space.is_square()
        A = Qu.random(space.shape, space.valence)
        A = A.dag() * A
        A = Gate.promote(A)
        return A

    @classmethod
    def random_unitary(cls, space):
        vs = []
        if type(space) is int:
            n = space
            space = Space((n, n), 'ud')
        assert space.is_square()
        n = space.shape[0]
        U = Gate(space.shape)
        for i in range(n):
            v = Qu.random(n, 'u')
            for u in vs:
                r = u.dag() * v
                v -= r * u
                #assert abs((u.transpose() * v)) < 1e-10
            v.normalize()
            vs.append(v)
            U[i,:] = v
        return U

    @classmethod
    def identity(cls, space):
        if type(space) is int:
            n = space
            space = Space((n, n), 'ud')
        A = cls(space.shape)
        assert space.is_square(), space
        for i in range(space.shape[0]):
            A[i, i] = 1.
        return A

    def is_hermitian(self, epsilon=EPSILON):
        assert self.space.is_square()
        B = self - ~self
        v = B.v.ravel()
        v = numpy.abs(v)
        r = numpy.sum(v)
        return r < epsilon

    def is_unitary(self, epsilon=EPSILON):
        assert self.space.is_square()
        A = self
        A = A * ~A
        I = self.identity(self.shape[0])
        v = A.v - I.v
        v = v.ravel()
        v = numpy.abs(v)
        r = numpy.sum(v)
        return r < epsilon

    def trace(A):
        # See also: Qu.contract
        assert A.space.is_square()
        r = zero
        for i in range(A.shape[0]):
            r += A.v[i, i]
        return r

    def is_pure(A, epsilon=EPSILON):
        assert A.space.is_square()
        A = A*A
        A = Gate.promote(A)
        return is_close(A.trace(), 1., epsilon)

    def expm(A, order=10):
        assert A.space.is_square()
        import scipy.linalg
        A = A.v 
        A = scipy.linalg.expm(A, order)
        return Gate(A.shape, A)

    def evolution_operator(H, t, order=10):
        assert H.space.is_square()
        import scipy.linalg
        float(t) # typecheck

        H = H.v 
        A = -1.j*t*H
        U = scipy.linalg.expm(A, order)
        return Gate(H.shape, U)
    get_U = evolution_operator




def build():
    # IDENTITY
    I = Gate.identity(2)
    I.name = 'I'
    Gate.I = I

    # FLIP
    X = Gate((2, 2))
    X[0, 1] = 1.
    X[1, 0] = 1.
    X.name = 'X'
    Gate.X = X

    Z = Gate((2, 2))
    Z[0, 0] = 1.
    Z[1, 1] = -1.
    Z.name = 'Z'
    Gate.Z = Z

    # HADAMARD
    H = Gate((2, 2))
    H[0, 0] = 1.
    H[0, 1] = 1.
    H[1, 0] = 1.
    H[1, 1] = -1.
    H *= 1./math.sqrt(2)
    H.name = 'H'
    Gate.H = H

    # C-NOT
    CN = Qu((2, 2, 2, 2), 'udud')
    CN[0, 0, 0, 0] = 1.
    CN[0, 0, 1, 1] = 1.
    CN[1, 1, 1, 0] = 1.
    CN[1, 1, 0, 1] = 1.
    Gate.CN = CN

    # SWAP
    SWAP = Qu((2, 2, 2, 2), 'udud')
    SWAP[0, 0, 0, 0] = 1.
    SWAP[1, 0, 0, 1] = 1.
    SWAP[0, 1, 1, 0] = 1.
    SWAP[1, 1, 1, 1] = 1.
    Gate.SWAP = SWAP

    # basis
    v0, v1 = [1, 0], [0, 1]
    Gate.dyads = (Ket(v0)@Bra(v0), Ket(v1)@Bra(v1))
    # make a method ??

    if scalar == numpy.complex128:
        Y = Gate((2, 2))
        Y[0, 1] = -1.j
        Y[1, 0] = 1.j
        Y.name = 'Y'
        Gate.Y = Y

        # The S gate, A.K.A. the P gate
        Gate.S = bitvec(0) @ ~bitvec(0) + 1.j*bitvec(1)@~bitvec(1)
    
        # The T gate
        r = numpy.exp(1.j*numpy.pi/4)
        Gate.T = bitvec(0) @ ~bitvec(0) + r*bitvec(1)@~bitvec(1)


build()


def commutator(A, B):
    return (A*B) - (B*A)

def anticommutator(A, B):
    return (A*B) + (B*A)

#def is_close(a, b, epsilon=EPSILON):
#    return abs(a-b) < epsilon

def is_close(v0, v1, epsilon=EPSILON):
    #if type(v0) != type(v1):
    #    return False
    if isinstance(v0, Qu):
        v0 = v0.v
    if isinstance(v1, Qu):
        v1 = v1.v
    if type(v0) == numpy.ndarray:
        if v0.shape != v1.shape:
            return False
        a = numpy.abs(v0-v1)
        r = numpy.sum(a)
    else:
        r = abs(v0-v1)
    return r < epsilon


