#!/usr/bin/env python3

import sys
import math
from random import choice, randint, seed, shuffle
from time import time
from operator import mul
from functools import reduce

import numpy
from numpy import dot
import numpy.linalg

from qupy.abstract import genidx
from qupy.dense import is_close
from qupy.ldpc.tool import write
from qupy.ldpc.solve import dot2, zeros2, shortstr, shortstrx, span, array2
#from qupy.ldpc.ensemble import Ensemble, loglikely
from qupy.argv import argv

epsilon = 1e-10
r2 = math.sqrt(2)

scalar = numpy.float64

if 'int' in sys.argv:
    scalar = numpy.int32


def flatstr(A):
    s = []
    for idx in genidx(A.shape):
        s_idx = ''.join(str(i) for i in idx)
        s.append("%s:%s"%(s_idx, A[idx]))
    return ', '.join(s)

def identity(n):
    I = numpy.zeros((n, n))
    for i in range(n):
        I[i, i] = 1.
    return I


class MPS(object):
    def __init__(self, As, linkss=None):
        self.As = As = [A.view() for A in As]
        self.linkss = linkss
        self.n = len(self.As)
        # shape: (d, r0), (r0, d, r1), (r1, d, r2), ..., (rn, d)
        if len(As[0].shape)==2:
            As[0].shape = (1,)+As[0].shape
        d = As[0].shape[1]
        rs = [As[0].shape[2]]
        for i in range(1, self.n-1):
            A = As[i]
            assert A.shape[0] == rs[-1]
            assert A.shape[1] == d
            r = A.shape[2]
            rs.append(r)
        i += 1
        A = As[i]
        if len(A.shape)==2:
            A.shape = A.shape+(1,)
        assert A.shape[0] == rs[-1]
        assert A.shape[1] == d
        self.rs = rs
        self.d = d
        self.dtype = A.dtype

    def __str__(self):
        return "MPS(%s)"%(', '.join(str(A.shape) for A in self.As))
        #return "MPS(%s)"%(', '.join(str(r) for r in self.rs))

    def get_khi(self):
        return max(A.shape[0] for A in self.As)

    def to_net(self, linkss=None):
        As = [A.view() for A in self.As]
        As[0].shape = As[0].shape[1:]
        As[-1].shape = As[-1].shape[:2]
        net = TensorNetwork(As, linkss or self.linkss)
        return net

    def get_dense(self):
        As = self.As
        idx = 0
        A0 = As[0] # (1, d, r0)
        A0 = A0.view()
        A0.shape = A0.shape[1:]
        d = self.d
        while idx+2<self.n:
            A1 = As[idx+1]
            A1 = A1.transpose((1, 2, 0)) # (d, r1, r0)
            assert A1.shape[0] == d
            A0 = numpy.inner(A0, A1) # (d, d, r1)
            shape = list(A0.shape)
            shape = shape[:len(shape)-1]
            assert shape == [d]*len(shape), tuple(shape)
            idx += 1
        A1 = As[idx+1]
        A1 = A1.view()
        A1.shape = A1.shape[:2]
        A1 = A1.transpose() # (d, rn)
        assert A1.shape[0] == d
        A0 = numpy.inner(A0, A1) # (d, d, r1)
        assert A0.shape == (d,)*self.n, A0.shape
        return A0

    @classmethod
    def build(cls, x, khi=None, links=None):

        d = x.shape[0]
        n0 = len(x.shape)
        assert x.shape == (d,)*n0

        As = []
        rs = []
    
        psi = x.view()
        psi.shape = (d, d**(n0-1))

        U, s, Vh = numpy.linalg.svd(psi, False)

        if khi is not None:
            U = U[:, :khi]
            s = s[:khi]
            Vh = Vh[:khi, :]
    
        r0 = U.shape[1]
    
        U.shape = (d, r0)
    
        s.shape = s.shape+(1,)
        Vh = s*Vh
    
        As.append(U)
        rs.append(U.shape[1])
    
        n = n0-2

        while n>=1:
            psi = Vh
            psi = psi.copy()
            psi.shape = (r0*d, d**n)
        
            U, s, Vh = numpy.linalg.svd(psi, False)
            assert U.shape == (psi.shape[0], min(psi.shape))
            assert s.shape == (min(psi.shape),)
            assert Vh.shape == (min(psi.shape), psi.shape[1])
        
            if khi is not None:
                U = U[:, :khi]
                s = s[:khi]
                Vh = Vh[:khi, :]
    
            #print "MPS.build:", n, s

            #print "U:", U.shape,
            #print "s:", s.shape,
            #print "Vh:", Vh.shape
        
            r1 = U.shape[1]
            U.shape = (r0, d, r1)
        
            s.shape = s.shape+(1,)
            Vh = s*Vh
        
            As.append(U)
            rs.append(r1)
        
            r0 = r1
            n -= 1
    
        #print Vh.shape
        As.append(Vh)
    
        assert len(As) == n0
    
        mps = MPS(As)
        return mps

    @classmethod
    def random(cls, n, khi):
        d = 2
        A = numpy.random.normal(0., 1., (1, d, khi))
        As = [A]
        for i in range(n-2):
            A = numpy.random.normal(0., 1., (khi, d, khi))
            As.append(A)
        A = numpy.random.normal(0., 1., (khi, d, 1))
        As.append(A)
        return MPS(As)

    @classmethod
    def from_ensemble(cls, E, p, khi=None):
#        print
#        print shortstrx(E.S)

        n = E.n

        shape = (2,)*n
        assert n<=20
        x = numpy.zeros(shape, dtype=numpy.float64)
        for i in range(E.N):
            row = E.S[i]
            #print row
            w = row.sum()
            pval = p**w * (1-p)**(n-w)
            x[tuple(row)] = pval # ?

        #print x.shape

        mps = cls.build(x, khi)

        #print mps

        y = mps.get_dense()
        idx = reindex(n, y.argmax())
        mps.idx = idx
        #print "argmax:", idx

        #err = numpy.abs((x-y)).sum() / numpy.abs(y).sum()
        err = numpy.sqrt(((x-y)**2).sum()/(2**n))

        print("Error:", err)

        return mps

    def left_canonical(self):
        As, n = self.As, self.n
        assert self.d == 2

        As = list(As) # mutate!

        Bs = []
        for i in range(n):
            A = As[i]
            assert len(A.shape)==3
            assert A.shape[1]==self.d

            r, d, c = A.shape

            # Eq. (48)
            A1 = numpy.zeros((r*d, c), dtype=scalar)
            A1[:r, :] = A[:, 0, :]
            A1[r:, :] = A[:, 1, :]

            #result = numpy.linalg.qr(A1, 'economic')
            Q, R = numpy.linalg.qr(A1)
            #print "left_canonical", (r, d, c), Q.shape, R.shape
            m = min(c, 2*r)
            assert Q.shape == (2*r, m)
            assert R.shape == (m, c)

            B = numpy.zeros((r, d, m), dtype=scalar)
            B[:, 0, :] = Q[:r]
            B[:, 1, :] = Q[r:]
            Bs.append(B)

            if i+1<n:
                #A = As[i+1].copy()
                A1 = numpy.zeros((m, d, As[i+1].shape[2]))
                #print "left_canonical: dot", R.shape, A[:,0,:].shape
                A1[:, 0, :] = dot(R, As[i+1][:, 0, :])
                A1[:, 1, :] = dot(R, As[i+1][:, 1, :])
                #print "left_canonical: set", As[i+1].shape, i+1, A1.shape
                As[i+1] = A1 # mutate!
    
            else:
                gamma = R[0, 0]

        mps = MPS(Bs, self.linkss)

        return gamma, mps

    def check_lcf(self, idxs=None):
        As, n = self.As, self.n
        for i in idxs or list(range(n)):
            A = As[i]
            A0, A1 = A[:, 0, :], A[:, 1, :]
            I = dot(A0.transpose(), A0) + dot(A1.transpose(), A1)
            assert numpy.allclose(I, identity(I.shape[0]))

    def check_rcf(self, idxs=None):
        As, n = self.As, self.n
        for i in idxs or list(range(n)):
            A = As[i]
            A0, A1 = A[:, 0, :], A[:, 1, :]
            I = dot(A0, A0.transpose()) + dot(A1, A1.transpose())
            assert numpy.allclose(I, identity(I.shape[0]))

    def truncate(self, khi):
        """
        See: 
        "Efficient Algorithms for Maximum Likelihood Decoding in the Surface Code",
        Sergey Bravyi, Martin Suchara, Alexander Vargo
        http://arxiv.org/abs/1405.4883
        """
        gamma, mps = self.left_canonical()
        As, n = mps.As, mps.n
        i = n-1
        while i>1:

            #for j in range(i):
            #    A = As[j]
            #    A0, A1 = A[:, 0, :], A[:, 1, :]
            #    I = dot(A0.transpose(), A0) + dot(A1.transpose(), A1)
            #    assert numpy.allclose(I, identity(I.shape[0]))

            #for j in range(i+1, n):
            #    A = As[j]
            #    assert A.shape[0]<=khi
            #    assert A.shape[2]<=khi
            #    A0, A1 = A[:, 0, :], A[:, 1, :]
            #    I = dot(A0, A0.transpose()) + dot(A1, A1.transpose())
            #    assert numpy.allclose(I, identity(I.shape[0]))

            A = As[i]
            r, d, c = A.shape

            # Eq. (51)
            A1 = numpy.zeros((r, 2*c), dtype=scalar)
            A1[:, :c] = A[:, 0, :]
            A1[:, c:] = A[:, 1, :]

            U, s, Vh = numpy.linalg.svd(A1, False)
            V = Vh.transpose()

            assert U.shape[0] == r
            assert V.shape[0] == 2*c
            assert U.shape[1]==s.shape[0]==V.shape[1]

            # Eq. (57)
            U1 = U[:, :khi]
            s1 = s[:khi]
            V1 = V[:, :khi]

            khi1 = U1.shape[1]

            assert U1.shape[1] == s1.shape[0] == V1.shape[1]

            #assert numpy.allclose(dot(U1.transpose(), U1), identity(khi1))
            #assert numpy.allclose(dot(V1.transpose(), V1), identity(khi1))

            s1.shape = (1, khi1) # for broadcast

            A = As[i-1] # mutate this
            A0, A1 = A[:, 0, :], A[:, 1, :]
            #print
            #print A0.shape, A1.shape
            A0 = dot(A0, U1)*s1
            A1 = dot(A1, U1)*s1
            #A0 = dot(dot(A0, U1), S1)
            #A1 = dot(dot(A1, U1), S1)
            assert A0.shape==A1.shape
            A = numpy.zeros((A0.shape[0], 2, A1.shape[1]), dtype=scalar)
            A[:, 0, :] = A0
            A[:, 1, :] = A1
            As[i-1] = A

            A = As[i] # mutate this
            A = numpy.zeros((khi1, 2, c), dtype=scalar)
            A[:, 0, :] = V1[:c].transpose()
            A[:, 1, :] = V1[c:].transpose()
            As[i] = A

            i -= 1

        As[0] = As[0] * gamma
        mps = MPS(As, self.linkss)
        return mps


def test_mps():

    n = 7
    d = 2
    shape = (d,)*n

    for i in range(5):

        khi = 4

        x = numpy.random.normal(0., 1., shape)
        x = MPS.random(n, khi).get_dense()
        x /= ((x**2).sum()/(d**n))**0.5

        mps = MPS.build(x)
        x1 = mps.get_dense()
        assert is_close(x, x1)
        #print mps

        gamma, mps1 = mps.left_canonical()
        mps1.As[0] *= gamma
        x1 = mps1.get_dense()
        assert is_close(x, x1)

        mps1 = MPS.build(x, khi)
        assert mps1.get_khi()==khi
        x1 = mps1.get_dense()
        #assert is_close(x, x1)
        #print mps1

        mps2 = mps.truncate(khi)
        #print mps2
        assert mps2.get_khi()==khi
        x2 = mps2.get_dense()

        assert ((x1-x2)**2).sum() < 1e-20
        assert ((x-x2)**2).sum() < 1e-20
        #print "err to mps(khi):", ((x1-x2)**2).sum()
        #print "err to original:", ((x-x2)**2).sum()


def reindex(n, idx0):
    idx = [0]*n
    for i in range(n):
        if (1<<i)&idx0:
            idx[n-i-1] = 1
    return idx


class MPSEnsembleDecoder(object):
    def __init__(self, code, C=10):
        self.code = code
        self.C = C

    def decodeXX(self, p, err_op, argv=None, **kw):
        C = self.C
        code = self.code
        A = code.Hz
        n = code.n

        fail = False

        E = Ensemble(code.n, C)
        E.fromrand(p)

        b = dot2(A, err_op)

#        print
#        print shortstrx(A, err_op, b)

        if b.max()==0:
            return zeros2(code.n)

        N = E.N

        A1 = A[numpy.where(b)]

        write('\n')
        write(E.S.shape, nl=True)
        write("first pass:", nl=True)

        # First we solve the b[i]=0 equations
        for i in range(A.shape[0]):
            if b[i]:
                continue

            E.solve0(A[i])

            b1 = dot2(A1, E.S.transpose())
            for i in range(b1.shape[0]):
                if b1[i].sum()==0:
                    write("x%d "%i)

        E.uniqify()

        write("\nsecond pass:\n")
        # Now we solve the b[i]=1 equations
        for i in range(A.shape[0]):
            if b[i]==0:
                continue

            if not E.solve1(A[i]):
                #return # <----------- return
                fail = True
                continue

        khi = argv.get('khi', None)
        mps = MPS.from_ensemble(E, p, khi)
    
        write("\n")
        if E.S.shape[0] == 0:
            return # <----------- return

        gamma, mps1 = mps.left_canonical()
        mps1.check_lcf()

        a, b = mps.get_dense(), gamma*mps1.get_dense()
        print("lcf error:", ((a-b)**2).sum())

        mps1 = mps.truncate(khi)

        a, b = mps.get_dense(), mps1.get_dense()
        print("truncate error:", ((a-b)**2).sum())
    
        #print
        #print shortstrx(E.S)

        ws = E.S.sum(1)
        idx = ws.argmin()
        x = E.S[idx]

        ll = loglikely(p, n, x.sum())
        write("solution: w=%.4f log(p)=%.1f\n" % (1.*x.sum()/n, ll))

        if ll < -8: # -8 ?
            fail = True

#        return None if fail else x
        return None if fail else mps.idx

    def build_parity(self, net, links, value=None, max_weight=8):

#        print "build_parity", links, value
        w = len(links)

        if w<=max_weight:
            shape = (2,)*(w+1)
            link = tuple(links)
            links = links+[link]
            A = numpy.zeros(shape, dtype=scalar)
            for idx in genidx(shape):
                if sum(idx[:w])%2==idx[w]:
                    A[idx] = 1

        else:

            links0, links1 = links[:w/2], links[w/2:]
            link0 = self.build_parity(net, links0)
            link1 = self.build_parity(net, links1)
            link = (link0, link1)
            shape = (2,2,2)
            A = numpy.zeros(shape, dtype=scalar)
            for i in (0,1):
              for j in (0,1):
                A[i,j,(i+j)%2]=1
            links = [link0, link1, link]

        if value is not None:
            w = len(A.shape)-1
            idx = tuple([slice(None)]*w)+(value,)
            A = A[idx]
            links.pop(-1)

#        print "append:", A.shape, links
        net.append(A, links)

        if value is None:
            return links[-1]

    #def contract_qubit(self, net, max_weight):

    def do_mps(self, net, khi):
        net1 = TensorNetwork()
        for i in range(len(net)):
            A, links = net[i]
            if len(links)<=4:
                net1.append(A, links)
                continue # <--------- continue

            mps = MPS.build(A, khi)
            write('m')
            assert len(mps.As)==len(links)
            for j, link in enumerate(links):
                A = mps.As[j]
                links = [('m', i, j-1), link, ('m', i, j)]
                if j==0:
                    links.pop(0)
                    A = A.view()
                    A.shape = A.shape[1:]
                elif j==len(mps.As)-1:
                    links.pop(2)
                    A = A.view()
                    A.shape = A.shape[:2]
                net1.append(A, links)
        return net1

    def decode(self, p, err_op, argv=None, **kw):
        C = self.C
        code = self.code
        Hz = code.Hz
        n = code.n
        m = Hz.shape[0]

        err_op[:] = 0
        err_op[1] = 1

        b = dot2(Hz, err_op)

        print("syndrome:", shortstr(b))

        net = TensorNetwork()

        # stabilizer nodes
        for i in range(m):
            write('s')

            a = Hz[i]
            links = [(i, int(j)) for j in numpy.where(a)[0]]
            if 0:
                w = a.sum()
                shape = (2,)*w
                A = numpy.zeros(shape, dtype=scalar)
    
                for idx in genidx(shape):
                    if sum(idx)%2==b[i]:
                        A[idx] = 1.
    
                net.append(A, links)
            else:
                self.build_parity(net, links, b[i], 8)

        # qubit nodes
        for i in range(n):
            write('q')
            a = Hz[:, i]
            w = a.sum()
            shape = (2,)*w
            A = numpy.zeros(shape, dtype=scalar)

            A[(0,)*w] = 1.-p # qubit is off
            A[(1,)*w] = p # qubit is on

            links = [(int(j), i) for j in numpy.where(a)[0]]
            net.append(A, links)

        net.todot("net.dot")
        print()
        print(shortstr(code.Hz))
        print(net.get_links())
        idx = 1
        for link in net.get_links():
            if link[1] != idx:
                net.contract_2(link)
                write('C')

        print()
        net.dump()
        for A, _ in net:
            print(repr(A))

        return

        #net.contract_easy(net)

        while net.contract_upto(16) or net.contract_easy():
            pass

        #net.dump()
        print()
        print(shortstr(err_op))
        print('~'*n)

        x = shortstr(err_op)
        xidx = x.index('1')
        for A, links in net:
          for _ in range(10):
            s = ['.']*n
            i = A.argmax()
            idx = numpy.unravel_index(i, A.shape)
            for i, link in enumerate(links):
                if type(link[0])==int:
                    s[link[1]] = str(idx[i])
            if '1' in s:
                c = '*' if s[xidx]=='1' else ('X' if s[xidx]=='0' else '')
                if c:
                    print(''.join(s), A[idx], c)
            A[idx] = 0.

        links = net.get_links()
        links = [l for l in links if type(l[0]) is int]
        links.sort(key = lambda link : (link[1], link[0]))
        #print links

        return

        khi = 4
        net = self.do_mps(net, khi)

        #net.dump()

        net.todot("net.dot")

        self.contract_upto(net, 8)

        net.dump()

        #for i in range(10):
        #    link = choice(net.get_links())
        #    net.contract_2(link)

        return

        net.contract_all()

        print()
        print(net.value)
        

class SparseTensor(object):
    def __init__(self, shape, keys=None, values=None, nnz=None, dtype=None):
        self.shape = shape
        self.keys = keys
        self.values = values

    @classmethod
    def fromdense(cls, A):
        A1 = A.ravel()
        keys = A1.nonzero()[0]
        values = numpy.array([A1[key] for key in keys], dtype=A.dtype)
        keys = [numpy.unravel_index(key, A.shape) for key in keys]
        keys = numpy.array(keys, dtype=numpy.int32)
        A = cls(A.shape, keys, values)
        return A


def broadcast_shape(A, B):
    assert len(A) == len(B)
    n = len(A)
    C = []
    for i in range(n):
        if A[i]==1:
            C.append(B[i])
        elif B[i]==1:
            C.append(A[i])
        elif A[i]==B[i]:
            C.append(A[i])
        else:
            assert 0
    return tuple(C)


def diagonal_shape(A, axis1, axis2):
    assert A[axis1] == A[axis2]
    B = list(A)
    if axis1 > axis2:
        axis2, axis1 = axis1, axis2
    B.pop(axis2)
    B.pop(axis1)
    B.append(A[axis1])
    return tuple(B)


def sum_shape(A, axis):
    A = list(A)
    A.pop(axis)
    return tuple(A)


def tensordot_shape(A, B, axes):
    axs, bxs = axes
    assert len(axs)==len(bxs)
    assert len(axs)==1
    ax = axs[0]
    bx = bxs[0]
    A = list(A)
    A.pop(ax)
    B = list(B)
    B.pop(bx)
    return tuple(A+B)


class DummyNetwork(object):
    def __init__(self, shapes=[], linkss=[]):
        self.shapes = list(shapes)
        self.linkss = list(linkss)
        self.n = len(self.shapes)
        self.cost = 0
        assert self.check()

    def check(self):
        shapes = self.shapes
        linkss = self.linkss
        assert len(shapes)==len(linkss)==self.n
        for shape, links in self:
            assert len(shape)==len(links), (shape, links)
        for link in self.get_links():
            idxs = self.has_link(link)
            shape = [shapes[idx][linkss[idx].index(link)] for idx in idxs]
            assert len(set(shape))==1, shape
        return True

    def update_cost(self):
        cost = 0
        for shape, links in self:
            cost = max(cost, reduce(mul, shape, 1))
        self.cost = max(cost, self.cost)

    def __str__(self):
        return "DummyNetwork(%s)"%(
            ', '.join(str(len(shape)) for shape in self.shapes))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        shapes = self.shapes
        linkss = self.linkss
        return shapes[i], linkss[i]

    def get_links(self):
        links = []
        for _links in self.linkss:
            links += _links
        links = list(set(links))
        links.sort()
        return links

    def has_link(self, link): # TOO SLOW
        linkss = self.linkss
        idxs = [i for i in range(self.n) if link in linkss[i]]
        return idxs

    def cleanup_link(self, i, link, verbose=False):
        shapes = self.shapes
        linkss = self.linkss
        shape = shapes[i]
        links = linkss[i]
        while links.count(link)>=2:
            j = links.index(link)
            k = links.index(link, j+1)
            assert len(shape)==len(links)
            if verbose:
                print("cleanup_link", shape, links, j, k)
                #print flatstr(shape)
            shape = diagonal_shape(shape, axis1=j, axis2=k)
            #print flatstr(shape)
            links.pop(k)
            links.pop(j)
            links.append(link)
            shapes[i] = shape
            if verbose:
                print("\t", shape.shape, links)
                print(flatstr(shape))
            assert len(shape)==len(links)
            assert self.check()
        self.update_cost()

    def cleanup(self, verbose=False):
        shapes = self.shapes
        linkss = self.linkss
        for i in range(self.n):
            shape = shapes[i]
            links = linkss[i]
            for link in set(links):
                if links.count(link)>1:
                    self.cleanup_link(i, link, verbose=verbose)
        for link in self.get_links():
            self.cleanup_freelegs(link, verbose=verbose)
        for i in range(self.n):
            shape = shapes[i]
            links = linkss[i]
            assert len(set(links))==len(links)
        assert self.check()
        self.update_cost()

    def cleanup_freelegs(self, link, verbose=False):
        shapes = self.shapes
        linkss = self.linkss

        idxs = [i for i in range(self.n) if link in linkss[i]]

        if len(idxs)==1:

            if verbose:
                print()
                print("cleanup_freelegs", link)
                print("\tidxs:", idxs)

            idx = idxs[0]
            shape = shapes[idx]
            links = linkss[idx]

            if verbose:
                print("\tsum axis", links.index(link))
            assert links.count(link)==1
            shape = sum_shape(shape, links.index(link))
            #print flatstr(shape)
            links.remove(link)
            shapes[idx] = shape
            if verbose:
                print("\t", shape, links)

            assert self.check()
        self.update_cost()

    def contract_1(self, link, verbose=False):
        shapes = self.shapes
        linkss = self.linkss

        idxs = [i for i in range(self.n) if link in linkss[i]]

        while len(idxs)>=3:
            if verbose:
                print()
                print("contract_1", link)
                print("\tidxs:", idxs)

            A, B = shapes[idxs[-2]], shapes[idxs[-1]]
            assert A is not B
            links, minks = linkss[idxs[-2]], linkss[idxs[-1]]
            j, k = links.index(link), minks.index(link)
            # pad shape for broadcast
            #print "A:", flatstr(A), links, j
            #print "B:", flatstr(B), minks, k
            A, B = (
                A[:j] + (1,)*k + A[j:j+1] \
                + (1,)*(len(B)-k-1) + A[j+1:],
                (1,)*j + B + (1,)*(len(A)-j-1))
            #print A
            #print B
            assert len(A)==len(B)

            #A = A*B # broadcast (inplace??)
            A = broadcast_shape(A, B)
            links = links[:j] + minks + links[j+1:]
            #print "A:", flatstr(A), links
            assert len(A)==len(links)

            shapes[idxs[-2]] = A
            linkss[idxs[-2]] = links

            shapes.pop(idxs[-1])
            linkss.pop(idxs[-1])
            self.n -= 1

            idxs.pop(-1)

            self.cleanup(verbose)

        assert self.check()
        self.update_cost()

    def contract_2(self, link, cleanup=True, verbose=False):
        shapes = self.shapes
        linkss = self.linkss
        if verbose:
            print()
            print("contract_2", link)

        #print linkss
        #for links in linkss:
        #    try:
        #        #link in links
        #        link == links[0]
        #    except:
        #        print link, type(link[0][0])
        idxs = [i for i in range(self.n) if link in linkss[i]]

        assert len(idxs) in (0, 2)
        #assert len(idxs) <= 2
        if len(idxs)==2:
            A, B = shapes[idxs[-2]], shapes[idxs[-1]]
            assert A is not B
            links, minks = linkss[idxs[-2]], linkss[idxs[-1]]
            #print "tensordot", links, minks
            assert links.count(link)==1 # not necessary...?
            assert minks.count(link)==1 # not necessary...?

            C = tensordot_shape(A, B,
                    ([links.index(link)], [minks.index(link)]))

            links.remove(link)
            minks.remove(link)
            shapes[idxs[-2]] = C
            linkss[idxs[-2]] = links + minks
            assert len(C)==len(links+minks)

            shapes.pop(idxs[-1])
            linkss.pop(idxs[-1])
            self.n -= 1

            idxs.pop(-1)

            if cleanup:
                #print links, minks
                self.cleanup(verbose) # XXX too slow XXX

        assert self.check()
        self.update_cost()

    def contract_scalars(self, verbose=False):
        for shape in self.shapes:
            assert shape==()
        self.shapes = [()]
        self.linkss = [[]]
        self.n = 1
        if verbose:
            self.dump()
        self.update_cost()

    def contract_all(self, verbose=False, rand=False):
        if verbose:
            self.dump()
        links = self.get_links()
        if rand:
            shuffle(links)
        for link in links:
            #break
            if verbose:
                self.dump()
            self.contract_1(link, verbose=verbose)
        self.cleanup(verbose)
        links = self.get_links()
        if rand:
            shuffle(links)
        for link in links:
            if verbose:
                self.dump()
            self.contract_2(link, verbose=verbose)
            #print([A.shape for A in self.shapes])
        if verbose:
            self.dump()
        #if self.get_links():
        #    self.contract_all_slow(verbose)
        if len(self)>1:
            self.contract_scalars(verbose)
        assert len(self)==1



class TensorNetwork(object):
    def __init__(self, As=[], linkss=[]):
        self.As = list(As)
        self.linkss = list(linkss)
        self.n = len(self.As)
        self.mark = {}
        assert self.check()

    def check(self):
        As = self.As
        linkss = self.linkss
        assert len(As)==len(linkss)==self.n
        for A, links in self:
            assert len(A.shape)==len(links), ((A.shape), (links))
        for link in self.get_links():
            idxs = self.has_link(link)
            shape = [As[idx].shape[linkss[idx].index(link)] for idx in idxs]
            assert len(set(shape))==1, shape
        return True

    def clone(self):
        As = [A.copy() for A in self.As]
        linkss = [list(links) for links in self.linkss]
        return TensorNetwork(As, linkss)

    def __str__(self):
        return "TensorNetwork(%s)"%(
            ', '.join(str(len(A.shape)) for A in self.As))

    def get_dummy(self):
        shapes = [A.shape for A in self.As]
        linkss = [list(links) for links in self.linkss]
        return DummyNetwork(shapes, linkss)

    @property
    def value(self):
        assert self.n == 1
        A = self.As[0]
        assert A.shape == (), "not a scalar"
        v = A[()]
        return v

    def get_links(self):
        links = []
        for _links in self.linkss:
            links += _links
        links = list(set(links))
        links.sort(key = str)
        return links

    def freelinks(self):
        linkss = self.linkss
        links = []
        for link in self.get_links():
            if sum(1 for links in linkss if links.count(link))>1:
                links.append(link)
        return links

    def neighbours(self, idx):
        linkss = self.linkss
        links = linkss[idx]
        nbd = []
        for link in links:
            for i in range(self.n):
                if link in linkss[i] and i!=idx:
                    nbd.append(i)
        return nbd

    def append(self, A, links):
        # links is a list of names for the "legs" of A
        assert len(A.shape)==len(links)
        assert len(set(links))==len(links), links # unique
        self.As.append(A)
        self.linkss.append(list(links)) # copy
        self.n += 1

    def pop(self, i):
        A = self.As.pop(i)
        links = self.linkss.pop(i)
        self.n -= 1
        return A, links

    def dump(self):
        As = self.As
        linkss = self.linkss
        print("TensorNetwork:")
        for i in range(self.n):
            print('\t%d'%i, As[i].shape, linkss[i])
            #print('\t', flatstr(As[i]), linkss[i])
            print(As[i])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        As = self.As
        linkss = self.linkss
        return As[i], linkss[i]

    def __setitem__(self, i, xxx_todo_changeme):
        (A, links) = xxx_todo_changeme
        self.As[i] = A
        self.linkss[i] = links

    def todot(self, name):
        As = self.As
        linkss = self.linkss
        #all = self.get_links()
        f = open(name, 'w')
        print("graph the_graph\n{", file=f)
        for i in range(len(self)):
            A = As[i]
            s = str(A.shape)
            s = s.replace(' ', '')
            print('\t A_%d [label="%s", shape=box];'%(i, s), file=f)
        for link in self.get_links():
            s = str(link)
            s = s.replace(' ', '')
            print('\t L_%s [shape=ellipse, label="%s"];'%(id(link), s), file=f)
            #print >>f, '\t L_%d_%d [shape=ellipse, label="%s"];'%(link[0], link[1], s)
            #print >>f, '\t L_%d_%d [shape=ellipse, label="L"];'%(link[0], link[1])
        for link in self.get_links():
            idxs = [i for i in range(self.n) if link in linkss[i]]
            for idx in idxs:
#                print >>f, "\t A_%d -- L_%d_%d;" % (idx, link[0], link[1])
                print("\t A_%d -- L_%s;" % (idx, id(link)), file=f)
        print("}", file=f)
        f.close()

    def select(self, idxs):
        As = self.As
        linkss = self.linkss
        As = [As[idx] for idx in idxs]
        linkss = [linkss[idx] for idx in idxs]
        return TensorNetwork(As, linkss)

    def paste(self, idxs, net):
        assert len(idxs)==len(net)
        for i, idx in enumerate(idxs):
            self[idx] = net[i]

    def has_link(self, link): # TOO SLOW
        linkss = self.linkss
        idxs = [i for i in range(self.n) if link in linkss[i]]
        return idxs

    def transpose(self, idx, perm):
        assert len(perm)==len(self.As[idx].shape)
        self.As[idx] = self.As[idx].transpose(perm)
        links = self.linkss[idx]
        links = [links[i] for i in perm]
        self.linkss[idx] = links

    def to_mps(self):
        As = self.As
        linkss = self.linkss
        assert self.n>=2

        #print self.linkss

        links = linkss[0]
        assert len(links)==2
        idxs = self.has_link(links[0])
        if len(idxs)>1:
            self.transpose(0, [1, 0])

        #print self.linkss
        links = linkss[0]
        assert len(self.has_link(links[0]))==1, self.has_link(links[0])
        assert len(self.has_link(links[1]))==2

        for i in range(1, self.n-1):
            links = linkss[i]
            assert len(links)==3
            if i-1 in self.has_link(links[1]):
                perm = [1]
            elif i-1 in self.has_link(links[2]):
                perm = [2]
            else:
                assert i-1 in self.has_link(links[0])
                perm = [0]

            if len(self.has_link(links[1]))==1:
                perm.append(1)
            elif len(self.has_link(links[2]))==1:
                perm.append(2)
            else:
                assert len(self.has_link(links[0]))==1
                perm.append(0)

            if i+1 in self.has_link(links[1]):
                perm.append(1)
            elif i+1 in self.has_link(links[2]):
                perm.append(2)
            else:
                assert i+1 in self.has_link(links[0])
                perm.append(0)

            assert sum(perm)==3
            if perm!=[0,1,2]:
                self.transpose(i, perm)

        i = self.n-1
        links = linkss[i]
        assert len(links)==2
        idxs = self.has_link(links[1])
        if len(idxs)>1:
            self.transpose(i, [1, 0])
        links = linkss[i]
        assert len(self.has_link(links[0]))==2
        assert len(self.has_link(links[1]))==1

        mps = MPS(As, linkss)
        #print mps, linkss
        #mps.to_net().check()
        return mps

    def shrink_at(self, idxs, link0, link1, verbose=False):
        As = self.As
        linkss = self.linkss
        assert len(idxs)==2
        assert link0!=link1

#        print "shrink_at", idxs, link0, link1

        idx0, idx1 = idxs
        A0, A1 = As[idx0], As[idx1]
        links0, links1 = linkss[idx0], linkss[idx1]

        #self.check()
        i0 = links0.index(link0)
        j0 = links0.index(link1)
        perm = list(range(len(links0)))
        perm[0], perm[i0] = perm[i0], perm[0]
        perm[1], perm[j0] = perm[j0], perm[1]
        self.transpose(idx0, perm)
#        print "transpose", idx0, perm
#        self.dump()
        As[idx0] = As[idx0].copy()
        #print A0.shape, linkss[idx0]
        As[idx0].shape = (As[idx0].shape[0]*As[idx0].shape[1],)+As[idx0].shape[2:]
        linkss[idx0].remove(link1)
        #print A0.shape, linkss[idx0]

        i1 = links1.index(link0)
        j1 = links1.index(link1)
        perm = list(range(len(links1)))
        perm[0], perm[i1] = perm[i1], perm[0]
        perm[1], perm[j1] = perm[j1], perm[1]
        self.transpose(idx1, perm)
#        print "transpose", idx1, perm
#        self.dump()
        As[idx1] = As[idx1].copy()
        As[idx1].shape = (As[idx1].shape[0]*As[idx1].shape[1],)+As[idx1].shape[2:]
        linkss[idx1].remove(link1)

#        self.dump()
        assert self.check()

    def shrink(self, verbose=False): # TOO SLOW
        "combine two legs into one"
        links = {}
        for link in self.get_links():
            idxs = tuple(self.has_link(link))
            if links.get(idxs):
                self.shrink_at(idxs, links[idxs], link, verbose=verbose)
            else:
                links[idxs] = link

    def cleanup_link(self, i, link, verbose=False):
        As = self.As
        linkss = self.linkss
        A = As[i]
        links = linkss[i]
        while links.count(link)>=2:
            j = links.index(link)
            k = links.index(link, j+1)
            assert len(A.shape)==len(links)
            if verbose:
                print("cleanup_link", A.shape, links, j, k)
                #print flatstr(A)
            #print("diagonal:", A.shape, end=" ")
            shape = diagonal_shape(A.shape, j, k)
            A = A.diagonal(axis1=j, axis2=k)
            #print(" =", A.shape)
            assert A.shape == shape
            #print flatstr(A)
            links.pop(k)
            links.pop(j)
            links.append(link)
            As[i] = A
            if verbose:
                print("\t", A.shape, links)
                print(flatstr(A))
            assert len(A.shape)==len(links)
            assert self.check()

    def cleanup(self, verbose=False):
        As = self.As
        linkss = self.linkss
        for i in range(self.n):
            A = As[i]
            links = linkss[i]
            for link in set(links):
                if links.count(link)>1:
                    self.cleanup_link(i, link, verbose=verbose)
        for link in self.get_links():
            self.cleanup_freelegs(link, verbose=verbose)
        for i in range(self.n):
            A = As[i]
            links = linkss[i]
            assert len(set(links))==len(links)
        assert self.check()

    def cleanup_freelegs(self, link, verbose=False):
        As = self.As
        linkss = self.linkss

        idxs = [i for i in range(self.n) if link in linkss[i]]

        if len(idxs)==1:

            if verbose:
                print()
                print("cleanup_freelegs", link)
                print("\tidxs:", idxs)

            idx = idxs[0]
            A = As[idx]
            links = linkss[idx]

            if verbose:
                print("\tsum axis", links.index(link))
            assert links.count(link)==1
            A = A.sum(links.index(link))
            #print flatstr(A)
            links.remove(link)
            As[idx] = A
            if verbose:
                print("\t", A.shape, links)

            assert self.check()

    def contract_1(self, link, verbose=False):
        As = self.As
        linkss = self.linkss

        idxs = [i for i in range(self.n) if link in linkss[i]]

        while len(idxs)>=3:
            if verbose:
                print()
                print("contract_1", link)
                print("\tidxs:", idxs)

            A, B = As[idxs[-2]], As[idxs[-1]]
            assert A is not B
            links, minks = linkss[idxs[-2]], linkss[idxs[-1]]
            j, k = links.index(link), minks.index(link)
            # pad shape for broadcast
            #print "A:", flatstr(A), links, j
            #print "B:", flatstr(B), minks, k
            A.shape, B.shape = (
                A.shape[:j] + (1,)*k + A.shape[j:j+1] \
                + (1,)*(len(B.shape)-k-1) + A.shape[j+1:],
                (1,)*j + B.shape + (1,)*(len(A.shape)-j-1))
            #print A.shape
            #print B.shape
            assert len(A.shape)==len(B.shape)

            #print("broadcast", A.shape, B.shape, end=" ")
            A = A*B # broadcast (inplace??)
            #print(" = ", A.shape)
            links = links[:j] + minks + links[j+1:]
            #print "A:", flatstr(A), links
            assert len(A.shape)==len(links)

            As[idxs[-2]] = A
            linkss[idxs[-2]] = links

            As.pop(idxs[-1])
            linkss.pop(idxs[-1])
            self.n -= 1

            idxs.pop(-1)

            self.cleanup(verbose)

        assert self.check()

    def contract_2(self, link, cleanup=True, verbose=False):
        As = self.As
        linkss = self.linkss
        if verbose:
            print()
            print("contract_2", link)

        #print linkss
        #for links in linkss:
        #    try:
        #        #link in links
        #        link == links[0]
        #    except:
        #        print link, type(link[0][0])
        idxs = [i for i in range(self.n) if link in linkss[i]]

        assert len(idxs) in (0, 2)
        #assert len(idxs) <= 2
        if len(idxs)==2:
            A, B = As[idxs[-2]], As[idxs[-1]]
            assert A is not B
            links, minks = linkss[idxs[-2]], linkss[idxs[-1]]
            #print "tensordot", links, minks
            assert links.count(link)==1 # not necessary...?
            assert minks.count(link)==1 # not necessary...?

            if 0:
                C = numpy.outer(A, B)
                C.shape = A.shape + B.shape
    
                As[idxs[-2]] = C
                linkss[idxs[-2]] = links + minks
                assert len(C.shape)==len(links+minks)
    
                As.pop(idxs[-1])
                linkss.pop(idxs[-1])
                self.n -= 1
    
                idxs.pop(-1)
    
                self.cleanup(verbose)
    
                return # <--------

            try:
                C = numpy.tensordot(A, B,
                    ([links.index(link)], [minks.index(link)]))
            except MemoryError:
                print("\ntensordot fail:", len(A.shape), "+", len(B.shape))
                raise

            links.remove(link)
            minks.remove(link)
            As[idxs[-2]] = C
            linkss[idxs[-2]] = links + minks
            assert len(C.shape)==len(links+minks)

            As.pop(idxs[-1])
            linkss.pop(idxs[-1])
            self.n -= 1

            idxs.pop(-1)

            if cleanup:
                #print links, minks
                self.cleanup(verbose) # XXX too slow XXX

        assert self.check()

    def contract_scalars(self, verbose=False):
        r = 1.
        for A in self.As:
            assert A.shape==()
            r *= A[()]
        self.As = [scalar(r)]
        self.linkss = [[]]
        self.n = 1
        if verbose:
            self.dump()

    def contract_easy(self):
        didit = False
        while 1:
            links = []
            for i in range(len(self)):
                if len(self.linkss[i])<=2:
                    link = self.linkss[i][0]
                    links.append(link)
            if not links:
                break
    
            #print "links", len(links)
            for link in links:
                self.contract_2(link, False)
                write('c')
            didit = True

        for i in range(len(self)):
            assert len(self.linkss[i])>2
        return didit

    #def contract_shortest(self):

    def contract_upto(self, max_weight, verbose=False):
        didit = False
        while 1:
            for link in self.get_links():
                idxs = self.has_link(link)
                w = sum(len(self.As[idx].shape) for idx in idxs)
                if w <= max_weight:
                    break
            else:
                break
            self.contract_2(link, verbose=verbose)
            write('u')
            didit = True
        return didit

    def contract_all(self, verbose=False):
        if verbose:
            self.dump()
        links = self.get_links()
        for link in links:
            #if link in skip:
            #    continue
            #break
            if verbose:
                self.dump()
            self.contract_1(link, verbose=verbose)
        self.cleanup(verbose)
        links = self.get_links()
        for link in links:
            #if link in skip:
            #    continue
            if verbose:
                self.dump()
            self.contract_2(link, verbose=verbose)
            #print([A.shape for A in self.As])
        if verbose:
            self.dump()
        #if self.get_links():
        #    self.contract_all_slow(verbose)
        if len(self)>1:
            self.contract_scalars(verbose)
        assert len(self)==1

    def contract_slow(self, link, verbose=False):
        As = self.As
        linkss = self.linkss
        idxs = [i for i in range(self.n) if link in linkss[i]]
        if verbose:
            print("contract_slow", link, idxs)
        if not idxs:
            return
        As = [A for A, links in self if link in links]
        linkss = [links for A, links in self if link in links]
        links = list(linkss[0])
        A = As[0]
        i = 0
        while i+1<len(As):
            A = numpy.outer(As[i], As[i+1])
            A.shape = As[i].shape + As[i+1].shape
            As[i+1] = A
            links += linkss[i+1]
            i += 1

        assert len(A.shape)==len(links)

        if verbose:
            print(A.shape, links)
        m = len(links)
        assert A.shape==(2,)*m
        links1 = [link1 for link1 in links if link1!=link]
        shape = (2,)*len(links1)
        B = numpy.zeros(shape, dtype=scalar)
        for idx in genidx((2,)*m):
            idx1 = list(idx)
            ii = None
            for i in reversed(list(range(m))):
                if links[i]==link:
                    j = idx1.pop(i)
                    if ii is None:
                        ii = j
                    elif ii!=j:
                        break
            else:
                idx1 = tuple(idx1)
                B[idx1] += A[idx]
        links = links1
        A = B

        for idx in reversed(idxs):
            self.As.pop(idx)
            self.linkss.pop(idx)
        self.As.insert(idx, A)
        self.linkss.insert(idx, links)
        self.n = len(self.As)

        assert self.check()

    def contract_all_slow(self, verbose=False):
        if verbose:
            self.dump()
        links = self.get_links()
        for link in links:
            if verbose:
                self.dump()
            self.contract_slow(link, verbose=verbose)
        if len(self)>1:
            self.contract_scalars(verbose)

    def contract_all_slow_slow(self, verbose=False):
        As = list(self.As)
        linkss = self.linkss
        links = list(linkss[0])
        i = 0
        while i+1<len(As):
            A = numpy.outer(As[i], As[i+1])
            A.shape = As[i].shape + As[i+1].shape
            As[i+1] = A
            links += linkss[i+1]
            i += 1

        while links:
            assert len(A.shape)==len(links)
            print(A.shape, links)
            #link = links[0]
            m = len(links)
            assert A.shape==(2,)*m
            links1 = [link for link in links if link!=links[0]]
            shape = (2,)*len(links1)
            B = numpy.zeros(shape, dtype=scalar)
            for idx in genidx((2,)*m):
                idx1 = list(idx)
                ii = None
                for i in reversed(list(range(m))):
                    if links[i]==links[0]:
                        j = idx1.pop(i)
                        if ii is None:
                            ii = j
                        elif ii!=j:
                            break
                else:
                    idx1 = tuple(idx1)
                    B[idx1] += A[idx]
            links = links1
            A = B

        self.As = [A]
        self.links = links
        self.n = 1

    def cost(self, link):
        "cost of contracting this link"
        idxs = self.has_link(link)
        cost = 0
        for idx in idxs:
            cost += len(self.As[idx].shape)
        return cost

    def paint(self, link, count=0):
        linkss = self.linkss
        mark = self.mark
        c = mark.get(link, count+1)
        if c <= count:
            return
        mark[link] = count
        for idx in self.has_link(link):
            links = linkss[idx]
            for link in links:
                self.paint(link, count+1)

    def get(self, idxs):
        As = self.As
        linkss = self.linkss
        r = 1.
        n = len(As)
        for i in range(n):
            A = As[i]
            links = linkss[i]
            idx = tuple(idxs[link] for link in links)
            v = A[idx]
            if v==0.:
                return 0.
            r *= v
        return r

    def contract_oe(self):
        import opt_einsum as oe
        args = []
        str_args = []
        for A, links in self:
            args.append(A)
            args.append(links)
            links = ''.join(oe.get_symbol(i) for i in links)
            str_args.append(links)
        str_args = ','.join(str_args)

        path, path_info = oe.contract_path(str_args, *self.As)
        #print(path_info)
        sz = path_info.largest_intermediate
        #print("(size: %d)" % sz, end="")

        if sz > 4194304:
            assert 0

        v = oe.contract(*args)
        #print("contract_oe", v.shape)
        assert v.shape == ()
        return v[()]



def build_test():

    if 0:
        n = 3
        k = 2
        shape = (2,)*k
        As = [numpy.random.normal(0., 1., shape) for i in range(n)]
        #As = [2.*numpy.random.binomial(1, 0.5, shape) for i in range(n)]
        #linkss = [range(k) for i in range(n)]
        linkss = [[0, 1], [1, 2], [2, 0]]

    elif 0:
        n = 2
        k = 3
        shape = (2,)*k
        As = [numpy.random.normal(0., 1., shape) for i in range(n)]
        #As = [2.*numpy.random.binomial(1, 0.5, shape) for i in range(n)]
        linkss = [list(range(k)) for i in range(n)]
        #linkss = [[0, 1], [1, 2], [2, 0]]

    elif 0:
        n = 1
        k = 4
        shape = (2,)*k
        As = [numpy.random.normal(0., 1., shape) for i in range(n)]
        #As = [2.*numpy.random.binomial(1, 0.5, shape) for i in range(n)]
        linkss = [list(range(k/2))*2]

    n = randint(1, 5)
    As = []
    linkss = []
    for i in range(n):
        k = randint(1, 5)
        shape = (2,)*k
        A = numpy.random.normal(0., 1., shape)
        links = [randint(0, 2*n) for _ in range(k)]
        As.append(A)
        linkss.append(links)

    net = TensorNetwork(As, linkss)
    return net


def test_net():

    #numpy.random.seed(0)
    #seed(0)

    #verbose = True
    verbose = False

    for trial in range(100):

        net = build_test()
        net1 = net.clone()
        #net2 = net.clone()

        idx = randint(0, net.n-1)
        perm = list(range(len(net.linkss[idx])))
        shuffle(perm)
        net.transpose(idx, perm)
    
        net.contract_all(verbose=verbose)
        #net.contract_2(0, verbose=verbose)
        #net.cleanup(verbose)
    
        net.contract_all_slow(verbose=verbose)
    
        #print "~~~~~~~~~~~~~~~~~\n"
    
        net1.contract_all_slow(verbose=verbose)
        #net2.contract_all_slow_slow(verbose=verbose)

        #net1.dump()
    
        #print net.value
        #print net1.value
        #print net2.value
    
        assert abs(net.value-net1.value)<1e-8
        write('.')
    print("OK")


def test_gauge():

    from qupy.ldpc.gcolor import Lattice

    n = int(sys.argv[1])
    lattice = Lattice(n)
    print(lattice)
    code = lattice.build_code()

    print(code.weightsummary())
    print(code.weightstr())

    #print shortstr(code.Gx)

    Gz, Gx = code.Gz, code.Gx
    mz, n = Gz.shape
    mx, _ = Gx.shape

    net = TensorNetwork()

    # Gx nodes flip qubits
    for i in range(mx):
        write('x')

        g = Gx[i]
        links = [('x', i, int(j)) for j in numpy.where(g)[0]]
        w = g.sum()
        shape = (2,)*w
        A = numpy.zeros(shape, dtype=scalar)

        # delta
        A[(0,)*w] = 1.
        A[(1,)*w] = 1.

        net.append(A, links)

    # qubit nodes: sum all bitflips
    for i in range(n):
        write('q')
        g = Gx[:, i]
        links = [('x', int(j), i) for j in numpy.where(g)[0]]
        links.append(i)
        w = g.sum()
        shape = (2,)*w
        A = numpy.zeros(shape+(2,), dtype=scalar)

        for idx in genidx(shape):
            idx = idx + (sum(idx)%2,)
            A[idx] = 1.

        net.append(A, links)

    # phase nodes
    for i in range(mz):
        write('z')

        g = Gz[i]
        links = [('z', i, int(j)) for j in numpy.where(g)[0]]
        w = g.sum()
        shape = (2,)*w
        A = numpy.zeros(shape, dtype=scalar)

        for idx in genidx(shape):
            c = sum(idx)%2
            A[idx] = -1. if c else 1.

        net.append(A, links)

    for i in range(n):
        write('q')
        g = Gz[:, i]
        links = [('z', int(j), i) for j in numpy.where(g)[0]]
        links.append(i)
        w = g.sum()+1
        shape = (2,)*w
        A = numpy.zeros(shape, dtype=scalar)

        # delta
        A[(0,)*w] = 1.
        A[(1,)*w] = 1.

        net.append(A, links)

    net.dump()

    corners = []
    for i in range(n):
        g = Gx[:, i]
        if g.sum()==3:
            corners.append(i)
    print("corners:", corners)

    for i in corners:
        i = mx + i
        links = net.linkss[i]
        print("paint:", links)
        for link in links:
            net.paint(link)
        break

    mark = net.mark
    links = list(mark.keys())
    links.sort(key = lambda link : mark[link])
    print([(link, mark[link]) for link in links])
    values = list(mark.values())
    values = list(set(values))
    values.sort()

    #for link in links:
    for value in reversed(values):
        links = [link for link in mark if mark[link]==value]
        while links:
            links.sort(key = net.cost)
            link = links.pop(0)
            c = net.cost(link)
            if c:
                print("contract_2:", c, value)
            if c<=22:
                net.contract_2(link)
                if c:
                    print(net, len(net))

    net.todot('net.dot')

    for i in range(len(net)):
        n = reduce(mul, net.As[i].shape)
        A = numpy.abs(net.As[i])
        print(i, n, A.sum())

        B = SparseTensor.fromdense(A)

    return

    net.contract_easy()
    net.contract_upto(8)
    net.contract_upto(12)
    net.contract_upto(16)
    for w in range(20, 35):
        net.contract_upto(w)
        print(net, len(net))
    #net.contract_all()

    print()
    #net.dump()
    print(net, len(net))


class ExactDecoder(object):
    "Tensor network decoder. Computes exact probabilities."
    "See OEDecoder for a faster version (smarter contractions.)"
    def __init__(self, code):
        self.code = code
        assert code.k <= 24, "too big...?"
        self.logops = list(span(code.Lx))

    def get_p(self, p, op, verbose=False):
        code = self.code
        Hz = code.Hz
        Tx = code.Tx
        Hx = code.Hx
        Lx = code.Lx
        n = code.n
        mx = code.mx

        net = TensorNetwork()

        # one tensor for each qubit
        for i in range(n):
            h = Hx[:, i]
            w = h.sum()
            assert w<20, "ouch"
            shape = (2,)*w
            A = numpy.zeros(shape, dtype=scalar)
            links = numpy.where(h)[0]

            opi = op[i]

            for idx in genidx(shape):
                if sum(idx)%2 == opi:
                    A[idx] = 1.-p # qubit is off
                else:
                    A[idx] = p # qubit is on

            net.append(A, links)

        net.contract_all(verbose)
        #net.contract_all_slow(verbose)

        return net.value

    def get_p_slow(self, p, op, verbose=False):
        code = self.code
        Hz = code.Hz
        Tx = code.Tx
        Hx = code.Hx
        Lx = code.Lx
        n = code.n
        mx = code.mx

        r = 0
        for idx in genidx((2,)*mx):
            h = op.copy()
            for i in range(mx):
                if idx[i]:
                    h += Hx[i]
            h %= 2
            w = h.sum()
            r += (p**w)*((1.-p)**(n-w))

        return r

    def decode(self, p, err_op, argv=None, verbose=False, **kw):
        code = self.code
        Hz = code.Hz
        Tx = code.Tx
        Hx = code.Hx
        Lx = code.Lx
        n = code.n
        mx = code.mx

        T = code.get_T(err_op)

        #if T.sum()==0:
        #    return T

        dist = []

        #print
        best = None
        best_r = 0.
        for logop in self.logops:
            op = (T+logop)%2
            r = self.get_p(p, op, verbose=verbose)
            #r1 = self.get_p_slow(p, op, verbose=verbose)
            #print "%.6f"%r, "%.6f"%r1
            if r>best_r:
                best_r = r
                best = op
            dist.append(r)

        #print(dist)
        return best


class OEDecoder(ExactDecoder):
    "Faster version of ExactDecoder "
    def __init__(self, code):
        self.code = code
        assert code.k <= 24, "too big...?"
        self.logops = list(span(code.Lx))
        self.n = code.n
        self.build()

    def build(self):
        import opt_einsum as oe

        #from opt_einsum.backends.dispatch import _has_einsum
        #_has_einsum['numpy'] = False

        code = self.code
        #Hz = code.Hz
        #Tx = code.Tx
        Hx = code.Hx
        #Lx = code.Lx
        n = code.n
        mx = code.mx

        net = []
        As = []
        linkss = []

        # one tensor for each qubit
        for i in range(n):
            h = Hx[:, i]
            w = h.sum()
            assert w<20, "ouch: w=%d"%w
            shape = (2,)*w
            A = numpy.zeros(shape, dtype=scalar)
            As.append(A)
            links = list(numpy.where(h)[0])
            linkss.append(links)
            net.append((A, links))
        #print(linkss)

        self.net = net
        self.linkss = linkss
        self.As = As
        self.check()

        kw = {"optimize" : "random-greedy"}

        str_args = []
        shapes = []
        for A, links in net:
            links = ''.join(oe.get_symbol(i) for i in links)
            str_args.append(links)
            shapes.append(A.shape)
        #print(shapes)
        #print(linkss)
        str_args = ','.join(str_args)
        #print(str_args)
        path, path_info = oe.contract_path(str_args, *As, **kw)
        #print(path_info)
        sz = path_info.largest_intermediate
        print("OEDecoder: size=%d" % sz)

#        if sz > 4194304:
        if sz > 134217728:
            assert 0, "too big... maybe"

        self.do_contract = oe.contract_expression(str_args, *shapes, **kw)

    def get_links(self):
        links = []
        for _links in self.linkss:
            links += _links
        links = list(set(links))
        links.sort()
        return links

    def has_link(self, link): # TOO SLOW
        linkss = self.linkss
        idxs = [i for i in range(self.n) if link in linkss[i]]
        return idxs

    def check(self):
        As = self.As
        linkss = self.linkss
        assert len(As)==len(linkss)
        for A, links in self.net:
            assert len(A.shape)==len(links), ((A.shape), (links))
        for link in self.get_links():
            idxs = self.has_link(link)
            shape = [As[idx].shape[linkss[idx].index(link)] for idx in idxs]
            assert len(set(shape))==1, shape
        return True

    do_contract = None
    def contract_oe(self):
        if self.do_contract is None:
            import opt_einsum as oe
            args = []
            for A, links in self.net:
                args.append(A)
                args.append(links)
                links = ''.join(oe.get_symbol(i) for i in links)
    
            v = oe.contract(*args)
            #print("contract_oe", v.shape)

        else:
            v = self.do_contract(*self.As)

        assert v.shape == ()
        return v[()]

    t0 = 0.
    t1 = 0.
    def get_p(self, p, op, verbose=False):
        code = self.code
        Hx = code.Hx
        n = code.n
        mx = code.mx

        t0 = time()
        
        # one tensor for each qubit
        for i in range(n):
            h = Hx[:, i]
            w = h.sum()
            assert w<20, "ouch"
            shape = (2,)*w
            A, links = self.net[i]
            opi = op[i]

            for idx in genidx(shape):
                if sum(idx)%2 == opi:
                    A[idx] = 1.-p # qubit is off
                else:
                    A[idx] = p # qubit is on

        t1 = time()

        value = self.contract_oe()

        t2 = time()

        self.t0 += t1-t0
        self.t1 += t2-t1

        #write(".")
        return value

    def fini(self):
        print("\nOEDecoder.t0 =", self.t0)
        print("OEDecoder.t1 =", self.t1)



class MPSDecoder(ExactDecoder):
    def __init__(self, code, khi=4):
        self.code = code
        assert code.k <= 24, "too big...?"
        self.logops = list(span(code.Lx))
        self.khi = khi
        print("khi:", khi)

    def get_p(self, p, op, verbose=False):
        code = self.code
        Hz = code.Hz
        Tx = code.Tx
        Hx = code.Hx
        Lx = code.Lx
        n = code.n
        mx = code.mx

        #print "build"
        net = TensorNetwork()

        # one tensor for each qubit
        for i in range(n):
            h = Hx[:, i]
            w = h.sum()
            assert w<20, "ouch"
            shape = (2,)*w
            A = numpy.zeros(shape, dtype=scalar)
            links = [(j,i) for j in numpy.where(h)[0]]

            opi = op[i]

            for idx in genidx(shape):
                if sum(idx)%2 == opi:
                    A[idx] = 1.-p # qubit is off
                else:
                    A[idx] = p # qubit is on

            net.append(A, links)

        # one tensor for each stabilizer
        for j in range(mx):
            h = Hx[j, :]
            w = h.sum()
            assert w<20, "ouch"
            shape = (2,)*w
            A = numpy.zeros(shape, dtype=scalar)
            links = [(j,i) for i in numpy.where(h)[0]]

            A[(0,)*w] = 1.
            A[(1,)*w] = 1.

            net.append(A, links)

        #net1 = net.clone()
        net1 = None

        #print("net.dot")
        #net.todot("net.dot")

        links = []
        for i in range(code.l):

            idx = code.keymap[i, 0, 0]
            assert len(net.linkss[idx])==1
            link = net.linkss[idx][0]
            links.append(link)
            idx = code.keymap[i, code.l-1, 0]
            assert len(net.linkss[idx])==1
            link = net.linkss[idx][0]
            links.append(link)

            for j in range(1, code.l-1):
                idx = code.keymap[i, j, 0]
                assert len(net.linkss[idx])==2
                link = net.linkss[idx][0]
                links.append(link)

        for i in range(code.l-1):
            for j in range(code.l-1):
                idx = code.keymap[i, j, 1]
                assert len(net.linkss[idx])==2
                link = net.linkss[idx][0]
                links.append(link)

        #print "links", len(links)
        for link in links:
            net.contract_2(link, False)

#        net.todot("net1.dot")
        #print "corners"

        corners = [idx for idx in range(len(net)) if len(net.linkss[idx])==2]
        assert len(corners)==4

        idx0 = corners[0]
        edge = [idx0]
        done = False
        while not done:
            idxs = net.neighbours(edge[-1])
            for i in idxs:
                sz = len(net.As[i].shape)
                if sz<=3 and i not in edge:
                    edge.append(i)
                    if sz==2:
                        done = True
                    break

        mps = net.select(edge)
        links = []
        for link in mps.get_links():
            if len(mps.has_link(link))==1:
                links.append(link)

        #mps.todot("net2.dot")
        #mps = mps.to_mps()

        for link in links:
            net.contract_2(link, False)

        #net.todot("net3.dot")

        net.shrink() # TOO SLOW

#        net.todot("net3.dot")

        while len(net)>code.l:
    
            corners = [idx for idx in range(len(net)) if len(net.linkss[idx])==2]
            assert len(corners)==4
    
            for idx in corners:
                A = net.As[idx]
                if max(A.shape)>2:
                    break
            else:
                assert 0
    
            edge = [idx]
            done = False
            while not done:
                idxs = net.neighbours(edge[-1])
                for i in idxs:
                    shape = net.As[i].shape
                    if len(shape)<=3 and i not in edge and max(shape)>2:
                        edge.append(i)
                        if len(shape)==2:
                            done = True
                        break
    
            mps = net.select(edge)
            links = []
            for link in mps.get_links():
                if len(mps.has_link(link))==1:
                    links.append(link)
    
            #mps.todot("net2.dot")
            mps = mps.to_mps()
            #print mps
            #print mps#, mps.linkss
            if mps.get_khi()>self.khi:
                #print "khi", mps.get_khi(), "->", self.khi
                mps = mps.truncate(self.khi)
                #print "khi", mps.get_khi()
            #print mps#, mps.linkss
            #print
            mps = mps.to_net()
            net.paste(edge, mps)
    
            for link in links:
                net.contract_2(link, False)
    
#            net.todot("net4.dot")
    
            net.shrink() # TOO SLOW
    
#            net.todot("net5.dot")
            write('.')

        #net.dump()
        net.contract_all(verbose)
        #print "OK"
        #net.contract_all_slow(verbose)

        if net1:
            net1.contract_all()
            assert abs(net.value-net1.value)<1e-6
            print(net.value, net1.value)

        return net.value


class BigLogopDecoder(object):
    """
        An exact decoder that builds a tensor network
        that represents the distribution over the logical operators.
    """

    def __init__(self, code):
        self.code = code
        assert code.k <= 24, "too big...?"

    def get_dist(self, p, op, verbose=False):
        code = self.code
        Hz = code.Hz
        Tx = code.Tx
        Hx = code.Hx
        Lx = code.Lx
        n = code.n
        k = code.k
        mx = code.mx

        HLx = numpy.concatenate((Hx, Lx))

        #print()
        #print(HLx)

        #print "build"
        net = []

        # one tensor for each qubit
        for i in range(n):
            h = HLx[:, i]
            w = h.sum()
            assert w<20, "ouch"
            shape = (2,)*w
            A = numpy.zeros(shape, dtype=scalar)
            links = [(j, i) for j in numpy.where(h)[0]]

            opi = op[i]

            for idx in genidx(shape):
                if sum(idx)%2 == opi:
                    A[idx] = 1.-p # qubit is off
                else:
                    A[idx] = p # qubit is on

            net.append((A, links))

        contract = []

        # one tensor for each stabilizer
        for j in range(mx):
            h = HLx[j, :]
            w = h.sum()
            assert w<20, "ouch"
            shape = (2,)*w
            A = numpy.zeros(shape, dtype=scalar)
            links = [(j, i) for i in numpy.where(h)[0]]

            A[(0,)*w] = 1.
            A[(1,)*w] = 1.

            net.append((A, links))
            contract += links

        # one tensor for each logop
        free = []
        for j in range(mx, mx+k):
            h = HLx[j, :]
            w = h.sum()+1 # add one free leg
            assert w<20, "ouch"
            shape = (2,)*w
            A = numpy.zeros(shape, dtype=scalar)
            links = [(j, i) for i in numpy.where(h)[0]]
            link = "l%d"%(j-mx)
            links.append(link)
            free.append(link)
            
            #print("logop", links[-1])

            A[(0,)*w] = 1.
            A[(1,)*w] = 1.

            net.append((A, links))
            contract += links

        contract = list(set(contract))

        all_links = []
        for A, links in net:
            assert len(links)==len(A.shape)
            all_links += links
        #for link in all_links:
        #    print(all_links.count(link), end=" ")
        #print()
        all_links = list(set(all_links))
        all_links.sort(key = str)
        lookup = dict((link, idx) for (idx, link) in enumerate(all_links))

        As = [A for (A, links) in net]
        linkss = [links for (A, links) in net]

        if 0:
            tn = TensorNetwork(As, linkss)
            #tn.dump()
            total = 0.
            idxs = {}
            for vec in genidx((2,)*len(all_links)):
                for i, link in enumerate(all_links):
                    idxs[link] = vec[i]
                val = tn.get(idxs)
                #if val > 0.:
                #    assert idxs['l0'] == 0
                #    assert idxs['l1'] == 0
                #    print("val:", val)
                if idxs['l0'] == 1 and idxs['l1'] == 0:
                    total += val
            print("total:", total)
    
            #tn = tn.clone()
            #tn.todot("net.dot")
            #tn.contract_all(skip=free)
            #print(tn.As)
            assert 0

        #print("links:", len(all_links))

        import opt_einsum as oe
        args = []
        str_args = []
        for A, links in net:
            args.append(A)
            links = ''.join(oe.get_symbol(lookup[i]) for i in links)
            #print(A.shape, links)
            args.append(links)
            str_args.append(links)
            assert len(links) == len(A.shape)

        free = ''.join(oe.get_symbol(lookup[i]) for i in free)
        args.append(free)
        str_args = ','.join(str_args) + '->' + free

        #v = oe.contract(*args)
        #print(str_args)
        path, path_info = oe.contract_path(str_args, *As)
        #print(path_info)
        sz = path_info.largest_intermediate
        print("OEDecoder: size=%d" % sz)

        if sz>4194304:
            assert 0, "ugh, too big"

        v = oe.contract(str_args, *As)
        #print("contract_oe", v.shape)

        return v

    def decode(self, p, err_op, argv=None, verbose=False, **kw):
        code = self.code
        Hz = code.Hz
        Tx = code.Tx
        Hx = code.Hx
        Lx = code.Lx
        n = code.n
        mx = code.mx

        T = code.get_T(err_op)

        #if T.sum()==0:
        #    return T

        dist = self.get_dist(p, T, verbose=verbose)

        # XXX find max in dist 

        print("...NOT FINISHED...")

        return T



class LogopDecoder(object):
    """
        An exact decoder that builds a tensor network
        that represents the distribution over the logical operators.
        This one uses less tensors than BigLogopDecoder.
    """

    def __init__(self, code):
        self.code = code
        assert code.k <= 26, "too big...?"

    def get_dist(self, p, op, verbose=False):
        code = self.code
        Hz = code.Hz
        Tx = code.Tx
        Hx = code.Hx
        Lx = code.Lx
        n = code.n
        k = code.k
        mx = code.mx

        HLx = numpy.concatenate((Hx, Lx))

        #print()
        #print(HLx)

        #print "build"
        net = []
        #free = list(range(mx, mx+k))
        free = []

        # one tensor for each qubit
        for i in range(n):
            h = HLx[:, i] # all the incident check operators
            w = h.sum()
            assert w<24, "ouch"
            shape = (2,)*w
            A = numpy.zeros(shape, dtype=scalar)
            links = [j for j in numpy.where(h)[0]]

            opi = op[i]

            for idx in genidx(shape):
                if sum(idx)%2 == opi:
                    A[idx] = 1.-p # qubit is off
                else:
                    A[idx] = p # qubit is on

            net.append((A, links))

        for j in range(mx, mx+k):
            A = numpy.zeros((2,2), dtype=scalar)
            A[0, 0] = 1.
            A[1, 1] = 1.
            idx = j+k
            free.append(idx)
            net.append((A, [j, idx]))

        all_links = []
        for A, links in net:
            assert len(links)==len(A.shape)
            all_links += links
        #for link in all_links:
        #    print(all_links.count(link), end=" ")
        #print()
        all_links = list(set(all_links))
        all_links.sort(key = str)
        lookup = dict((link, idx) for (idx, link) in enumerate(all_links))

        As = [A for (A, links) in net]
        linkss = [links for (A, links) in net]

        if 0:
            tn = TensorNetwork(As, linkss)
            #tn.dump()
            total = 0.
            idxs = {}
            for vec in genidx((2,)*len(all_links)):
                for i, link in enumerate(all_links):
                    idxs[link] = vec[i]
                val = tn.get(idxs)
                #if val > 0.:
                #    assert idxs['l0'] == 0
                #    assert idxs['l1'] == 0
                #    print("val:", val)
                if idxs['l0'] == 1 and idxs['l1'] == 0:
                    total += val
            print("total:", total)
    
            #tn = tn.clone()
            #tn.todot("net.dot")
            #tn.contract_all(skip=free)
            #print(tn.As)
            assert 0

        #print("links:", len(all_links))

        import opt_einsum as oe
        args = []
        str_args = []
        for A, links in net:
            args.append(A)
            links = ''.join(oe.get_symbol(lookup[i]) for i in links)
            #print(A.shape, links)
            args.append(links)
            str_args.append(links)
            assert len(links) == len(A.shape)

        free = ''.join(oe.get_symbol(lookup[i]) for i in free)
        args.append(free)
        str_args = ','.join(str_args) + '->' + free

        #v = oe.contract(*args)
        #print(str_args)
        path, path_info = oe.contract_path(str_args, *As)
        #print(path_info)
        sz = path_info.largest_intermediate
        print("(size=%d)" % (sz,), end='', flush=True)

#        if sz>33554432:
        if sz>268435456:
            assert 0, "ugh, too big"

        v = oe.contract(str_args, *As)
        #print("contract_oe", v.shape)

        return v

    def decode(self, p, err_op, argv=None, verbose=False, **kw):
        code = self.code
        Hz = code.Hz
        Tx = code.Tx
        Hx = code.Hx
        Lx = code.Lx
        n = code.n
        mx = code.mx

        T = code.get_T(err_op)

        #if T.sum()==0:
        #    return T

        dist = self.get_dist(p, T, verbose=verbose)

        #print(dist.shape)
        #print(dist)

        best_idx = None
        best_v = 0.
        for idx in genidx(dist.shape):
            val = dist[idx]
            if val > best_v:
                best_idx = idx
                best_v = val

        #print("best:", best_idx)

        best_idx = array2(best_idx)
        op = dot2(best_idx, Lx)
        op = (T+op) % 2

        return op





if __name__=="__main__":

    test_mps()
    test_net()
    #test_gauge()



