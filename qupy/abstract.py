#!/usr/bin/env python3

import sys
import math

from operator import mul

#import numpy
#import numpy.random
#
#scalar = numpy.complex128
#zero = scalar(0.)


def genidx(shape):
    if len(shape)==0:
        yield ()
    else:
        for idx in range(shape[0]):
            for _idx in genidx(shape[1:]):
                yield (idx,)+_idx


class Space(object):
    def __init__(self, shape, valence):
        if type(shape) is int:
            shape = (shape,)
        assert len(shape) == len(valence)
        self.shape = tuple(shape)
        self.valence = valence

    def __matmul__(self, other):
        shape = self.shape + other.shape
        valence = self.valence + other.valence
        return Space(shape, valence)

    def __eq__(self, other):
        return (self.shape, self.valence) == (other.shape, other.valence) 

    def __ne__(self, other):
        return (self.shape, self.valence) != (other.shape, other.valence) 

    def __str__(self):
        return "Space(%s, %r)"%(self.shape, self.valence)
    __repr__ = __str__

    def __getitem__(self, idx):
        return (self.shape, self.valence)[idx]

    def __hash__(self):
        return hash((self.shape, self.valence))

    def dual(self):
        valence = ''.join('u' if v=='d' else 'd' for v in self.valence)
        return Space(self.shape, valence)

    __invert__ = dual

    def unify(A, B):
        rank = len(A.shape)
        if rank != len(B.shape):
            return None # fail
        dim0 = [(A.shape[idx], A.valence[idx]) for idx in range(rank)]
        dim1 = [(B.shape[idx], B.valence[idx]) for idx in range(rank)]
        perm = []
        for n, v in dim0:
            if (n, v) in dim1:
                j = dim1.index((n, v))
                perm.append(j)
                dim1[j] = None # erase
            else:
                return None # fail
        perm = [perm.index(i) for i in range(len(perm))]
        return perm

#    def __mul__(self, other):
#        assert self.valence.count('d') == other.valence.count('u')
#        up_shape = [n for idx, n in enumerate(self.shape) if
#            self.valence[idx]=='u']
#        dn_shape = [n for idx, n in enumerate(other.shape) if
#            other.valence[idx]=='d']
#        shape = up_shape + dn_shape
#        valence = 'u'*len(up_shape) + 'd'*len(dn_shape)
#        return Space(shape, valence)

    def __mul__(A, B):
        i = 0
        j = 0
        rank0 = len(A.shape)
        rank1 = len(B.shape)
        output = []
        while i < rank0 or j < rank1:
            if i == rank0:
                assert B.valence[j] == 'd'
                output.append((j, B.shape[j], 'd'))
                j += 1
            elif j == rank1:
                assert A.valence[i] == 'u'
                output.append((i, A.shape[i], 'u'))
                i += 1
            elif A.valence[i] == 'u':
                output.append((i, A.shape[i], 'u'))
                i += 1
            elif B.valence[j] == 'd':
                output.append((j, B.shape[j], 'd'))
                j += 1
            else:
                assert A.valence[i] == 'd'
                assert B.valence[j] == 'u'
                assert A.shape[i] == B.shape[j], \
                    "shape mismatch, i=%d j=%d"%(i, j)
                # Contract this pair.
                i += 1
                j += 1
        output.sort(key = lambda item:item[0])
        shape = [item[1] for item in output]
        valence = [item[2] for item in output]
        return Space(shape, ''.join(valence))

    def is_square(self):
        return self.valence in ('ud', 'du') \
            and self.shape[0] == self.shape[1]



class AbstractQu(object):

    """ lower indices represent 'components' of covariant vectors (covectors), 
        while upper indices represent components of contravariant vectors (vectors).
    """

    def __init__(self, shape, valence):
        if type(shape) is int:
            shape = (shape,)
        assert len(shape) == len(valence)
        for updn in valence:
            assert updn in 'ud'
        assert type(valence) is str
        self.rank = len(shape)
        self.space = Space(shape, valence)
        #self.name = "%s(???)"%self.__class__.__name__
        self.name = None

    @property
    def shape(self):
        return self.space.shape

    def get_valence(self):
        return self.space.valence

    def set_valence(self, valence):
        self.space.valence = valence
    valence = property(get_valence, set_valence)

    def __str__(self):
        return "%s(%s, %s)"%(self.__class__.__name__, self.shape, self.valence)
    __repr__ = __str__

    def match_valence(self, other):
        dn_count = self.valence.count('d')
        up_count = other.valence.count('u')
        assert dn_count == up_count,\
            "cannot match valence %s * %s" % (self.valence, other.valence)

    def unify(A, B):
        if not isinstance(B, AbstractQu):
            raise ValueError
        perm = A.space.unify(B.space)
        if perm is None:
            return None
        C = A.permuted(perm)
        #if A.valence != B.valence:
        #    print "<<%s>>"%C.valence,
        assert C.space == B.space, "unify: %s, %s"%(A.space, B.space)
        return C

    def genidx(self):
        return genidx(self.shape)

    def __eq__(self, other):
        if self.space!=other.space:
            return False
        for idx in self.genidx():
            if self[idx] != other[idx]: # EPSILON ? symbolic?
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __mul__(self, other):
        self.match_valence(other)
        box = LazyMulQu(self, other)
        return box

    def dag(self):
        return LazyDagQu(self)

    def __invert__(self):
        return self.dag()

    def is_gate(self):
        # square too ?
        return list(self.valence).count('u') == list(self.valence).count('d')


class LazyMulQu(AbstractQu):
    def __init__(self, left, right):
        shape = left.shape
        valence = left.valence
        AbstractQu.__init__(self, shape, valence)
        self.left = left
        self.right = right

    def __mul__(self, other):
        self.match_valence(other)
        # Try to right associate. Possibly recursive:
        return self.left * (self.right * other)

    def inverse(self):
        return self.right.inverse() * self.left.inverse()

    def dag(self):
        return self.right.dag() * self.left.dag()


class LazyDagQu(AbstractQu):
    def __init__(self, child):
        shape = child.shape
        valence = ''.join('u' if v is 'd' else 'd' for v in child.valence)
        AbstractQu.__init__(self, shape, valence)

    def dag(self):
        return self.child
    


class IdentityQu(AbstractQu):
    def __init__(self, shape, valence):
        AbstractQu.__init__(self, shape, valence)

    def __mul__(self, other):
        self.match_valence(other)
        return other

    def inverse(self):
        return self
    dag = inverse


class PermutationQu(AbstractQu):
    def __init__(self, shape, valence, perm):
        AbstractQu.__init__(self, shape, valence)
        assert len(perm) == len(shape)
        assert set(perm) == set(range(self.rank))
#        valence = tuple(self.valence[i] for i in perm)
#        assert valence == self.valence
        self.perm = perm

    def __mul__(self, other):
        # XXX does this make sense ?????
        self.match_valence(other)
        if not isinstance(other, Qu): # ConcreteQu ?
            return AbstractQu.__mul__(self, other)

        valence = tuple(self.valence[i] for i in perm)



#    def inverse(self):




