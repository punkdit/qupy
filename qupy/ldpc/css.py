#!/usr/bin/env python3

import sys, os
from math import *
from random import *
import time

import numpy
import numpy.random as ra
from numpy.linalg import lstsq

write = lambda s : print(s, end='', flush=True)

if __name__ == "__main__":

    from qupy.argv import argv
    datestr = os.popen('date "+%F %H:%M:%S"').read().strip()


#from qupy.tool.smap import SMap
from qupy.ldpc import solve
from qupy.ldpc.solve import (
    shortstr, shortstrx, hbox,
    eq2, dot2, compose2, rand2,
    pop2, insert2, append2, pushout,
    array2, zeros2, identity2, parse)
#from qupy.ldpc.tool import write, load, save

from qupy.ldpc.decoder import Decoder, RandomDecoder
#from qupy.ldpc.dynamic import Tanner

from qupy.ldpc.chain import Chain, ChainMap

from qupy.ldpc.bpdecode import RadfordBPDecoder

from qupy.ldpc.cluster import ClusterCSSDecoder
#from qupy.ldpc import ensemble
#from qupy.ldpc import lp
#
#try:
#    from qupy.ldpc.metro import metropolis
#except ImportError:
#    print("metro module failed to load")

dot = numpy.dot


def sparsestr(A):
    idxs = numpy.where(A)[0]
    return "[[%s]]"%(' '.join(str(i) for i in idxs))


def check_conjugate(A, B):
    if A is None or B is None:
        return
    assert A.shape == B.shape
    I = numpy.identity(A.shape[0], dtype=numpy.int32)
    assert eq2(dot(A, B.transpose())%2, I)


def check_commute(A, B):
    if A is None or B is None:
        return
    C = dot2(A, B.transpose())
    assert C.sum() == 0, "\n%s"%shortstr(C)


def poprand(items):
    i = randint(0, len(items)-1)
    item = items.pop(i)
    return item


def direct_sum(A, B):
    if A is None or B is None:
        return None
    m = A.shape[0] + B.shape[0] # rows
    n = A.shape[1] + B.shape[1] # cols
    C = zeros2(m, n)
    C[:A.shape[0], :A.shape[1]] = A
    C[A.shape[0]:, A.shape[1]:] = B
    return C


class CSSCode(object):

    def __init__(self,
            Lx=None, Lz=None, 
            Hx=None, Tz=None, 
            Hz=None, Tx=None, # XXX swap these two args?
            Gx=None, Gz=None, 
            build=True,
            check=True, verbose=True):

        if Hx is None and Hz is not None:
            # This is a classical code
            Hx = zeros2(0, Hz.shape[1])

        self.Lx = Lx
        self.Lz = Lz
        self.Hx = Hx
        self.Tz = Tz
        self.Hz = Hz
        self.Tx = Tx
        self.Gx = Gx
        self.Gz = Gz

        if Hx is not None and len(Hx.shape)<2:
            Hx.shape = (0, Hz.shape[1])
        if Hz is not None and Hx is not None and Lz is not None and Gz is None:
            assert Hx.shape[0]+Hz.shape[0]+Lz.shape[0] == Hx.shape[1]
            assert Hx.shape[1] == Hz.shape[1] == Lz.shape[1], Lz.shape

        n = None
        if Gz is not None and Gx is not None:
            _, n = Gz.shape
            if build:
                self.build_from_gauge(check=check)
        #elif None in (Lx, Lz, Tx, Tz) and build:
        elif build and (Lx is None or Lz is None or Tx is None or Tz is None):
            self.build(check=check)
        elif Hz is not None and Hx is not None:
            _, n = Hz.shape
            self.k = n - Hz.shape[0] - Hx.shape[0]

        for op in [Lx, Lz, Hx, Tz, Hz, Tx]:
            if op is None:
                continue
            #print op.shape
            if n is not None and op.shape==(0,):
                op.shape = (0, n)
            n = op.shape[1]
        self.n = n
        if self.Hx is not None:
            self.mx = self.Hx.shape[0]
        if self.Hz is not None:
            self.mz = self.Hz.shape[0]
        if self.Lx is not None:
            self.k = self.Lx.shape[0]
        if self.Gx is not None:
            self.gx = self.Gx.shape[0]
        if self.Gz is not None:
            self.gz = self.Gz.shape[0]

        self.check = check
        self.do_check()

    def copy(self):
        Lx, Lz, Hx, Tz, Hz, Tx = (
            self.Lx, self.Lz, self.Hx,
            self.Tz, self.Hz, self.Tx)
        Lx = Lx.copy()
        Lz = Lz.copy()
        Hx = Hx.copy()
        Tz = Tz.copy()
        Hz = Hz.copy()
        Tx = Tx.copy()
        code = CSSCode(
            Lx=Lx, Lz=Lz, Hx=Hx, Tz=Tz, 
            Hz=Hz, Tx=Tx, check=self.check)
        return code

    def __add__(self, other):
        "perform direct sum"
        Gx, Gz = direct_sum(self.Gx, other.Gx), direct_sum(self.Gz, other.Gz)
        Hx, Hz = direct_sum(self.Hx, other.Hx), direct_sum(self.Hz, other.Hz)
        Lx, Lz = direct_sum(self.Lx, other.Lx), direct_sum(self.Lz, other.Lz)
        Tx, Tz = direct_sum(self.Tx, other.Tx), direct_sum(self.Tz, other.Tz)
        code = CSSCode(
            Lx=Lx, Lz=Lz, Hx=Hx, Tz=Tz, 
            Hz=Hz, Tx=Tx, check=self.check)
        return code

    def __rmul__(self, r):
        assert type(r) is int
        code = self
        for i in range(r-1):
            code = self + code
        return code

    def __hash__(self):
        ss = []
        for H in [
            self.Lx, self.Lz, self.Hx,
            self.Tz, self.Hz, self.Tx]:
            if H is not None:
                ss.append(H.tostring())
        return hash(tuple(ss))

    def build_from_gauge(self, check=True, verbose=False):

        write("build_from_gauge:")

        Gx, Gz = self.Gx, self.Gz
        Hx, Hz = self.Hx, self.Hz
        Lx, Lz = self.Lx, self.Lz

        #print "build_stab"
        #print shortstr(Gx)
        #vs = solve.find_kernel(Gx)
        #vs = list(vs)
        #print "kernel Gx:", len(vs)
    
        n = Gx.shape[1]
    
        if Hz is None:
            A = dot2(Gx, Gz.transpose())
            vs = solve.find_kernel(A)
            vs = list(vs)
            #print "kernel GxGz^T:", len(vs)
            Hz = zeros2(len(vs), n)
            for i, v in enumerate(vs):
                Hz[i] = dot2(v.transpose(), Gz) 
        Hz = solve.linear_independent(Hz)
    
        if Hx is None:
            A = dot2(Gz, Gx.transpose())
            vs = solve.find_kernel(A)
            vs = list(vs)
            Hx = zeros2(len(vs), n)
            for i, v in enumerate(vs):
                Hx[i] = dot2(v.transpose(), Gx)
        Hx = solve.linear_independent(Hx)

        if check:
            check_commute(Hz, Hx)
            check_commute(Hz, Gx)
            check_commute(Hx, Gz)

        Gxr = numpy.concatenate((Hx, Gx))
        Gxr = solve.linear_independent(Gxr)
        assert eq2(Gxr[:len(Hx)], Hx)
        Gxr = Gxr[len(Hx):]
    
        Gzr = numpy.concatenate((Hz, Gz))
        Gzr = solve.linear_independent(Gzr)
        assert eq2(Gzr[:len(Hz)], Hz)
        Gzr = Gzr[len(Hz):]

        if Lx is None:
            Lx = solve.find_logops(Gz, Hx)
        if Lz is None:
            Lz = solve.find_logops(Gx, Hz)

        write('\n')

        assert len(Gxr)==len(Gzr)
        kr = len(Gxr)
        V = dot2(Gxr, Gzr.transpose())
        U = solve.solve(V, identity2(kr))
        assert U is not None
        Gzr = dot2(U.transpose(), Gzr)

        if check:
            check_conjugate(Gxr, Gzr)
            check_commute(Hz, Gxr)
            check_commute(Hx, Gzr)
            check_commute(Lz, Gxr)
            check_commute(Lx, Gzr)

        assert len(Lx)+len(Hx)+len(Hz)+len(Gxr)==n

        self.Lx, self.Lz = Lx, Lz
        self.Hx, self.Hz = Hx, Hz
        self.Gxr, self.Gzr = Gxr, Gzr

    def build(self, check=True, verbose=False):
    
        Hx, Hz = self.Hx, self.Hz
        Lx, Lz = self.Lx, self.Lz
        #print shortstrx(Hx)
        #print "%d stabilizers"%len(Hx)
    
        #print shortstrx(Hx)
        #print shortstrx(solve.row_reduce(Hx))
        #print shortstrx(solve.row_reduce(Hx))

        write('li:')
        self.Hx = Hx = solve.linear_independent(Hx)
        self.Hz = Hz = solve.linear_independent(Hz)
    
        mz, n = Hz.shape
        mx, nx = Hx.shape
        assert n==nx
        assert mz+mx<=n, (mz, mx, n)
    
        write('build:')
        if verbose:
            print(shortstrx(Hx))
    
        if check:
            # check kernel of Hx contains image of Hz^t
            check_commute(Hx, Hz)
    
        if Lz is None:
            write('find_logops(Lz):')
            Lz = solve.find_logops(Hx, Hz, verbose=verbose)
            #print shortstr(Lz)
            #write(len(Lz))

        k = len(Lz)
        assert n-mz-mx==k, "_should be %d logops, found %d. Is Hx/z degenerate?"%(
            n-mx-mz, k)
        print("n=%d, mx=%d, mz=%d, k=%d" % (n, mx, mz, k))
    
        # Find Lx --------------------------
        if Lx is None:
            write('find_logops(Lx):')
            Lx = solve.find_logops(Hz, Hx, verbose=verbose)

        assert len(Lx)==k

        if check:
            check_commute(Lx, Hz)
            check_commute(Lz, Hx)


        U = dot2(Lz, Lx.transpose())
        I = identity2(k)
        A = solve.solve(U, I)
        assert A is not None
        #assert eq2(dot2(U, A), I)
        #assert eq2(dot2(Lz, Lx.transpose(), A), I)

        Lx = dot2(A.transpose(), Lx)

        if check:
            check_conjugate(Lz, Lx)

        # Find Tz --------------------------
        write('Find(Tz):')
        U = zeros2(mx+k, n)
        U[:mx] = Hx
        U[mx:] = Lx
        B = zeros2(mx+k, mx)
        B[:mx] = identity2(mx)

        Tz_t = solve.solve(U, B)
        Tz = Tz_t.transpose()
        assert len(Tz) == mx

        check_conjugate(Hx, Tz)
        check_commute(Lx, Tz)

        # Find Tx --------------------------
        write('Find(Tx):')
        U = zeros2(n, n)
        U[:mz] = Hz
        U[mz:mz+k] = Lz
        U[mz+k:] = Tz

        B = zeros2(n, mz)
        B[:mz] = identity2(mz)
        Tx_t = solve.solve(U, B)
        Tx = Tx_t.transpose()

        write('\n')
    
        check_conjugate(Hz, Tx)
        check_commute(Lz, Tx)
        check_commute(Tz, Tx)

        self.k = k
        self.Lx = Lx
        self.Lz = Lz
        self.Tz = Tz
        self.Tx = Tx

    def do_check(self):
        if not self.check:
            return
        write("checking...")
        Lx, Lz, Hx, Tz, Hz, Tx, Gx, Gz = (
            self.Lx, self.Lz, self.Hx,
            self.Tz, self.Hz, self.Tx,
            self.Gx, self.Gz)
        check_conjugate(Lx, Lz)
        check_conjugate(Hx, Tz)
        check_conjugate(Hz, Tx)
        #if Gx is not None and Gz is not None:
        #    check_conjugate(Gx, Gz)
        check_commute(Lx, Tz)
        check_commute(Lx, Hz)
        check_commute(Lz, Tx)
        check_commute(Lz, Hx)
        check_commute(Hx, Hz)
        check_commute(Tx, Tz)
        all_ops = [Lx, Lz, Hx, Tz, Hz, Tx]
        for ops in all_ops:
            if ops is not None:
                assert ops.shape[1] == self.n
        write("done\n")

    _dual = None
    def dual(self, build=False, check=False):
        if self._dual:
            return self._dual
        code = CSSCode(
            self.Lz, self.Lx, 
            self.Hz, self.Tx, 
            self.Hx, self.Tz, self.Gz, self.Gx, 
            build, self.check or check)
        self._dual = code
        return code

    def build_all(self):
        self.all_Lx = list(solve.span(self.Lx))
        #self.all_Lz = list(solve.span(self.Lz))
        self.all_Hx = list(solve.span(self.Hx))
        #self.all_Hz = list(solve.span(self.Hz))

    def __str__(self):
        Lx = len(self.Lx) if self.Lx is not None else None
        Lz = len(self.Lz) if self.Lz is not None else None
        Hx = len(self.Hx) if self.Hx is not None else None
        Tz = len(self.Tz) if self.Tz is not None else None
        Hz = len(self.Hz) if self.Hz is not None else None
        Tx = len(self.Tx) if self.Tx is not None else None
        Gx = len(self.Gx) if self.Gx is not None else None
        Gz = len(self.Gz) if self.Gz is not None else None
        n = self.n
        #if Lx is None and Hz and Hx:
        #    Lx = "(%d)"%(n - Hx - Hz)
        #if Lz is None and Hz and Hx:
        #    Lz = "(%d)"%(n - Hx - Hz)
        return "CSSCode(n=%s, Lx:%s, Lz:%s, Hx:%s, Tz:%s, Hz:%s, Tx:%s, Gx:%s, Gz:%s)" % (
            n, Lx, Lz, Hx, Tz, Hz, Tx, Gx, Gz)

    def save(self, name=None, stem=None):
        assert name or stem
        if stem:
            s = hex(abs(hash(self)))[2:]
            name = "%s_%s_%d_%d_%d.ldpc"%(stem, s, self.n, self.k, self.d)
            print("save", name)
        f = open(name, 'w')
        for name in 'Lx Lz Hx Tz Hz Tx Gx Gz'.split():
            value = getattr(self, name, None)
            if value is None:
                continue
            print("%s ="%name, file=f)
            print(shortstr(value), file=f)
        f.close()

    @classmethod
    def load(cls, name, build=True, check=False, rebuild=False):
        write("loading..")
        f = open(name)
        data = f.read()
        data = data.replace('.', '0')
        lines = data.split('\n')
        name = None
        items = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if '=' in line:
                name = line.split('=')[0].strip()
                #print name
                rows = []
                items[name] = rows
            else:
                #line = list(int(x) for x in line)
                line = numpy.fromstring(line, dtype=numpy.uint8) - 48
                rows.append(line)
        #print items.keys()
        #print [len(x) for x in items.values()]
        kw = {}
        for key in list(items.keys()):
            #print items[key]
            value = array2(items[key])
            kw[key] = value
        write("done\n")
        if rebuild:
            for op in 'Lx Lz Tx Tz'.split():
                kw[op] = None
        code = cls(build=build, check=check, **kw)
        return code

    def longstr(self):
        lines = [
            "CSSCode:",
            "Lx:Lz =", shortstrx(self.Lx, self.Lz),
            "Hx:Tz =", shortstrx(self.Hx, self.Tz),
            "Tx:Hz =", shortstrx(self.Tx, self.Hz)
        ]
        return '\n'.join(lines)

    def weightstr(self, logops=False):
        lines = [
            "Lx:%s" % [op.sum() for op in self.Lx] 
                if self.Lx is not None else "Lx:?",
            "Lz:%s" % [op.sum() for op in self.Lz] 
                if self.Lz is not None else "Lz:?",
            "Hx:%s" % [op.sum() for op in self.Hx] 
                if self.Hx is not None else "Hx:?",
            "Tz:%s" % [op.sum() for op in self.Tz] 
                if self.Tz is not None else "Tz:?",
            "Hz:%s" % [op.sum() for op in self.Hz] 
                if self.Hz is not None else "Hz:?",
            "Tx:%s" % [op.sum() for op in self.Tx] 
                if self.Tx is not None else "Tx:?"]
        if logops:
            lines = lines[:2]
        return '\n'.join(lines)

    def weightsummary(self):
        lines = []
        for name in 'Lx Lz Hx Tz Hz Tx'.split():
            ops = getattr(self, name)
            if ops is None:
                continue
            m, n = ops.shape
            rweights = [ops[i].sum() for i in range(m)]
            cweights = [ops[:, i].sum() for i in range(n)]
            if rweights:
                lines.append("%s(%d:%.0f:%d, %d:%.0f:%d)"%(
                    name, 
                    min(rweights), 1.*sum(rweights)/len(rweights), max(rweights),
                    min(cweights), 1.*sum(cweights)/len(cweights), max(cweights),
                ))
            else:
                lines.append("%s()"%name)
        return '\n'.join(lines)

    @property
    def distance(self):
        "simple upper bound on the distance"
        w = self.n
        Lx, Lz = self.Lx, self.Lz
        if Lx is not None:
            for op in Lx:
                w = min(op.sum(), w)
        if Lz is not None:
            for op in Lz:
                w = min(op.sum(), w)
        return w
    d = distance

    def get_chain(self):
        if self.mx and self.mz:
            chain = Chain([self.Hx, self.Hz.transpose()])
        elif self.mx:
            chain = Chain([self.Hx])
        else:
            chain = Chain([self.Hz])
        chain.check()
        return chain

    def product(self, other, build=False, check=False):
        c0 = self.get_chain()
        c1 = other.get_chain()

        chain = c0.tensor(c1)
        codes = chain.allcodes(build=build, check=check)
        return codes

    def prune_deadbits(self):
        "a dead bit lacks either x or z type stabilizer (or both)"
        n, mx, mz = self.n, self.mx, self.mz
        Hx, Hz = self.Hx.copy(), self.Hz.copy()
        alive = list(range(n))

        cx = Hx.sum(0)
        for idx, w in enumerate(cx):
            if w==0 and idx in alive:
                alive.pop(alive.index(idx))

        cz = Hz.sum(0)
        for idx, w in enumerate(cz):
            if w==0 and idx in alive:
                alive.pop(alive.index(idx))

        Hx = Hx[:, alive]
        Hz = Hz[:, alive]

        code = CSSCode(Hx=Hx, Hz=Hz)
        return code


#    def swap(self, i, j, n, mx, mz):
#        Lx, Lz, Hx, Tz, Hz, Tx = (
#            self.Lx, self.Lz, self.Hx,
#            self.Tz, self.Hz, self.Tx)
#        Ux = numpy.concatenate((Lx, Hx, Tx))
#        Uz = numpy.concatenate((Lz, Hz, Tz))

    def logop2hx(self, i=0, j=0):
        "move the i-th x type logical operator into Hx"
        Lx, Lz, Hx, Tz, Hz, Tx = (
            self.Lx, self.Lz, self.Hx,
            self.Tz, self.Hz, self.Tx)
        lx, lz = Lx[i], Lz[i]
        self.Lx, self.Lz = pop2(Lx, i), pop2(Lz, i)
        self.Hx, self.Tz = insert2(Hx, j, lx), insert2(Tz, j, lz)
        self.k -= 1
        self.mx += 1
        self.do_check()

    def logop2hz(self, i=0, j=0):
        "move the i-th z type logical operator into Hz"
        Lx, Lz, Hx, Tz, Hz, Tx = (
            self.Lx, self.Lz, self.Hx,
            self.Tz, self.Hz, self.Tx)
        lx, lz = Lx[i], Lz[i]
        self.Lx, self.Lz = pop2(Lx, i), pop2(Lz, i)
        self.Tx, self.Hz = insert2(Tx, j, lx), insert2(Hz, j, lz)
        self.k -= 1
        self.mz += 1
        self.do_check()

    def hx2logop(self, i=0, j=0):
        Lx, Lz, Hx, Tz, Hz, Tx = (
            self.Lx, self.Lz, self.Hx,
            self.Tz, self.Hz, self.Tx)
        opx, opz = Hx[i], Tz[i]
        self.Hx, self.Tz = pop2(Hx, i), pop2(Tz, i)
        self.Lx, self.Lz = insert2(Lx, j, opx), insert2(Lz, j, opz)
        self.k += 1
        self.mx -= 1
        self.do_check()

    def hz2logop(self, i=0, j=0):
        Lx, Lz, Hx, Tz, Hz, Tx = (
            self.Lx, self.Lz, self.Hx,
            self.Tz, self.Hz, self.Tx)
        opx, opz = Tx[i], Hz[i]
        self.Tx, self.Hz = pop2(Tx, i), pop2(Hz, i)
        self.Lx, self.Lz = insert2(Lx, j, opx), insert2(Lz, j, opz)
        self.k += 1
        self.mz -= 1
        self.do_check()

    def hx2hz(self, i=0, j=0):
        Lx, Lz, Hx, Tz, Hz, Tx = (
            self.Lx, self.Lz, self.Hx,
            self.Tz, self.Hz, self.Tx)
        opx, opz = Hx[i], Tz[i]
        self.Hx, self.Tz = pop2(Hx, i), pop2(Tz, i)
        self.Tx, self.Hz = insert2(Tx, j, opx), insert2(Hz, j, opz)
        self.mz += 1
        self.mx -= 1
        self.do_check()

    def hz2hx(self, i=0, j=0):
        Lx, Lz, Hx, Tz, Hz, Tx = (
            self.Lx, self.Lz, self.Hx,
            self.Tz, self.Hz, self.Tx)
        opx, opz = Tx[i], Hz[i]
        self.Tx, self.Hz = pop2(Tx, i), pop2(Hz, i)
        self.Hx, self.Tz = insert2(Hx, j, opx), insert2(Tz, j, opz)
        self.mz -= 1
        self.mx += 1
        self.do_check()

    def prune_xlogops(self, cutoff, verbose=False):

        # Prune x type logops
        Lx = self.Lx
        ws = [op.sum() for op in Lx]
        m, n = Lx.shape
        i = m-1
        while i>=0:
            if ws[i]<=cutoff:
                self.logop2hx(i)
                #if verbose:
                #    print self.longstrx(self)
            i -= 1

    def prune_logops(self, cutoff, verbose=False):

        # Prune x type logops
        Lx = self.Lx
        ws = [op.sum() for op in Lx]
        m, n = Lx.shape
        i = m-1
        while i>=0:
            if ws[i]<=cutoff:
                self.logop2hx(i)
                #if verbose:
                #    print self.longstrx(self)
            i -= 1

        # Prune z type logops
        Lz = self.Lz
        ws = [op.sum() for op in Lz]
        m, n = Lz.shape
        i = m-1
        while i>=0:
            if ws[i]<=cutoff:
                self.logop2hz(i)
            i -= 1

        # XX this is biased against z type logops XX

    def shorten(self, verbose=False):
        "use stabilizers to decrease weight of logical ops"

        # could also row reduce Lx then use lower ops to
        # kill higher ops... Um..

        Lx, Lz, Hx, Tz, Hz, Tx = (
            self.Lx, self.Lz, self.Hx,
            self.Tz, self.Hz, self.Tx)

        graph = Tanner(Hx)
        for i in range(Lx.shape[0]):
            #write("i=%d w=%d "%(i, Lx[i].sum()))
            op = graph.localmin(Lx[i], verbose=verbose)
            if op.sum() < Lx[i].sum():
                Lx[i] = op
                write('.')
            else:
                write(choice('\/'))

        graph = Tanner(Hz)
        for i in range(Lz.shape[0]):
            #write("i=%d w=%d "%(i, Lz[i].sum()))
            op = graph.localmin(Lz[i], verbose=verbose)
            if op.sum() < Lz[i].sum():
                Lz[i] = op
                write('.')
            else:
                write(choice('\/'))

        self.build()
        self.do_check()

    def find_distance(self, stopat=None):
        decoder = StarDynamicDistance(self)
        L = decoder.find(stopat=stopat, verbose=False)
        #print shortstr(L)
        #print "distance <=", L.sum()
        return L.sum()

    def get_T(self, err_op):
        # bitflip, x-type errors, hit Hz ops, produce Tx
        n = self.n
        T = zeros2(n)
        Hz = self.Hz
        Tx = self.Tx
        m = Hz.shape[0]
        for i in range(m):
            if dot2(err_op, Hz[i]):
                T += Tx[i]
        T %= 2
        return T

    def get_dist(self, p, T):
        "distribution over logical operators"
        dist = []
        sr = 0.
        n = self.n
        for l_op in self.all_Lx:
            #print "l_op:", shortstr(l_op)
            r = 0.
            T1 = l_op + T
            for s_op in self.all_Hx:
                T2 = (s_op + T1)%2
                d = T2.sum()
                #print shortstr(T2), d
                #print d,
                r += (1-p)**(n-d)*p**d
            #print
            sr += r
            dist.append(r)
        dist = [r//sr for r in dist]
        return dist

    def is_stab_0(self, v):
        write('/')
        Hx_t = self.Hx.transpose()
        u = solve.solve(Hx_t, v)
        if u is not None:
            #print "[%s]"%u.sum(),
            v1 = dot2(Hx_t, u)
            assert ((v+v1)%2).max() == 0
        write('\\')
        return u is not None

#    Hx_t_inv = None
    def is_stab(self, v):
        write('/')
        Hx_t = self.Hx.transpose()

#        # Hx_t u = v
#        # u = Hx_t^-1 v
#        if self.Hx_t_inv is None:
#            Hx_t_inv = solve.pseudo_inverse(Hx_t)
#            self.Hx_t_inv = Hx_t_inv
#        Hx_t_inv = self.Hx_t_inv
#        u = dot2(Hx_t_inv, v)

        u = dot2(self.Tz, v)

        #u = solve.solve(Hx_t, v)
        #if u is not None:
        if eq2(dot2(Hx_t, u), v):
            #print "[%s]"%u.sum(),
            v1 = dot2(Hx_t, u)
            assert ((v+v1)%2).max() == 0
#            assert self.is_stab_0(v) # double check
        else:
#            assert not self.is_stab_0(v) # double check
            u = None
        write('\\')
        return u is not None


def concat(Cout, Cin):
    n = Cout.n * Cin.n
    #print Cout.longstr()

    Hx = []
    for i in range(Cout.mx):
        Hout = Cout.Hx[i]
        for j in range(Cin.k):
            Lin = Cin.Lx[j]
            #print Hout, Lin
            h = numpy.tensordot(Hout, Lin, 0)
            #h = shortstr(h.flatten())
            h = h.flatten()
            #print h
            Hx.append(h)

    Hz = []
    for i in range(Cout.mz):
        Hout = Cout.Hz[i]
        for j in range(Cin.k):
            Lin = Cin.Lz[j]
            #print Hout, Lin
            h = numpy.tensordot(Hout, Lin, 0)
            h = h.flatten()
            #print h
            assert len(h) == n
            Hz.append(h)

    for i in range(Cout.n):
        for j in range(Cin.mx):
            h = zeros2(n)
            h[i*Cin.n : (i+1)*Cin.n] = Cin.Hx[j]
            Hx.append(h)
        for j in range(Cin.mz):
            h = zeros2(n)
            h[i*Cin.n : (i+1)*Cin.n] = Cin.Hz[j]
            Hz.append(h)

    #print Hx

    Hx = array2(Hx)
    Hz = array2(Hz)

    #print shortstr(Hx)

    C = CSSCode(Hx=Hx, Hz=Hz)
    return C


#class StarDynamicDistance(Decoder):
#    """ Search for small weight logical operators.
#        These upper-bound the distance.
#    """
#    def __init__(self, code):
#        self.code = code
#        Hx, Lx = self.Hx, self.Lx
#
#
#    # XX _should be method mindist of CSSCode ? XX
#    def find_full(self, target=None, stopat=None, verbose=False):
#        Hx = self.Hx
#        Lx = self.Lx
#
#        M0 = argv.get("M0", 10)
#        target = target or argv.target
#
#        best_T = Lx[0]
#        best_w = best_T.sum()
#        for i in range(Lx.shape[0]):
#        
#            H = zeros2(Lx.shape[0]-1+Hx.shape[0], self.n)
#            H[:i] = Lx[:i]
#            H[i:Lx.shape[0]-1] = Lx[i+1:]
#            H[Lx.shape[0]-1:] = Hx
#            graph = Tanner(H)
#
#            T1 = Lx[i].copy()
#            for j in range(M0):
#                #print "l_op:"
#                #print strop(l_op)
#                T1 = best_T.copy()
#                for op in Hx:
#                    if random()<0.5:
#                        T1 += op
#                        T1 %= 2
#                for op in Lx:
#                    if random()<0.5:
#                        T1 += op
#                        T1 %= 2
#                if target:
#                    T1 = graph.minimize(T1, target, maxsize=argv.maxsize, verbose=verbose)
#                    break
#                else:
#                    T1 = graph.localmin(T1, stopat=stopat, verbose=verbose)
#                w = T1.sum()
#                if w and w < best_w:
#                    best_T = T1
#                    best_w = w
#                    write("%s:"%w)
#                    if stopat is not None and w <= stopat:
#                        return best_T
#
#        return best_T
#
#    def find(self, target=None, stopat=None, verbose=False):
#        Hx = self.Hx
#        Lx = self.Lx
#
#        M0 = argv.get("M0", 1)
#        target = target or argv.target
#
#        best_i = 0
#        best_w = Lx[0].sum()
#        for i in range(1, Lx.shape[0]):
#            w = Lx[i].sum() 
#            if w < best_w:
#                best_i = i
#                best_w = w
#
#        i = best_i
#        best_T = Lx[i]
#        write("%s:"%best_w)
#
#        H = zeros2(Lx.shape[0]-1+Hx.shape[0], self.n)
#        H[:i] = Lx[:i]
#        H[i:Lx.shape[0]-1] = Lx[i+1:]
#        H[Lx.shape[0]-1:] = Hx
#        graph = Tanner(H)
#
#        T1 = Lx[i].copy()
#        for j in range(M0):
#            #print "l_op:"
#            #print strop(l_op)
#            T1 = best_T.copy()
#            #for op in Hx:
#            #    if random()<0.5:
#            #        T1 += op
#            #        T1 %= 2
#            #for op in Lx:
#            #    if random()<0.5:
#            #        T1 += op
#            #        T1 %= 2
#            if target:
#                T1 = graph.minimize(T1, target, maxsize=argv.maxsize, verbose=verbose)
#                break
#            else:
#                T1 = graph.localmin(T1, stopat=stopat, verbose=verbose)
#            w = T1.sum()
#            if w and w < best_w:
#                best_T = T1
#                best_w = w
#                write("%s:"%w)
#                if stopat is not None and w <= stopat:
#                    return best_T
#
#        return best_T


def build_tree(n):

    assert n%2

    Hz = []
    Hx = []
    k0 = -1
    k1 = 1
    while k1+1<n:

        h = zeros2(n)
        h[max(0, k0)] = 1
        h[k1] = 1
        h[k1+1] = 1
        Hz.append(h)

        if k0+2>=n//4:
            h = zeros2(n)
            h[k1] = 1
            h[k1+1] = 1
            Hx.append(h)

        k1 += 2
        k0 += 1

    Hz = array2(Hz)
    Hx = array2(Hx)

    print(shortstr(Hz))
    print()
    print(shortstr(Hx))

    code = CSSCode(Hz=Hz, Hx=Hx)
    return code


#from qupy.braid.tree import Tree
#class Graph(object):
#    def __init__(self, nbd={}):
#        self.nbd = dict(nbd)
#
#    def add_edge(self, a, b):
#        nbd = self.nbd
#        nbd.setdefault(a, []).append(b)
#        nbd.setdefault(b, []).append(a)
#
#    def metric(self, a, b):
#        tree = Tree(a)
#        nbd = self.nbd
#        path = tree.grow_path(nbd, b)
#        return len(path)


def build_cycle(m, n, row):

    H = []

    for i in range(n):
        h = [0]*n
        for j, c in enumerate(row):
            h[(i+j)%n] = int(c)
        H.append(h)

    delta = n-m
    for i in range(delta):
        H.pop(i*n//delta-i)

    Hz = array2(H)

    return Hz


def opticycle(m, n):

    best_d = 0

    for trial in range(4000):

        #row = '1101'
        row = [0] * n
        for i in range(18):
            row[randint(0, n-1)] = 1
        print(shortstr(row))

        Hz = build_cycle(m, n, row)

        Hz = solve.row_reduce(Hz, truncate=True)
    
        #print shortstr(Hz)
    
        code = CSSCode(Hz=Hz, build=True, check=True)
    
        Lx = code.Lx
        k = Lx.shape[0]
        ws = [Lx[i].sum() for i in range(k)]
        d = min(ws)
        print("\nd =", d)

        if d > best_d:
            d = code.find_distance(stopat=best_d)
            print("\n*d =", d)

        if d > best_d:
            best_d = d
            best = code
            print("distance <=", best_d)

    code = best

    print()
    print(code)
    #print code.weightstr()
    print(code.weightsummary())
    print("distance <=", best_d)

    return code


def random_distance_stab(code):

    Lx = code.Lx
    m, n = Lx.shape

    best_i = None
    best_w = n

    for i in range(m):
        w = Lx[i].sum() 
        if w < best_w:
            best_i = i
            best_w = w

    write("%d:"%best_w)
    L = Lx[best_i]

    ops = [Lx[i] for i in range(m) if i != best_i]\
        + [code.Hx[i] for i in range(code.Hx.shape[0])]

    L = L.copy()
    L1 = L.copy()

    while 1:

        op = choice(ops)
        L1 += op
        L1 %= 2
        w = L1.sum()
        if w <= best_w:
            L, L1 = L1, L
            if w < best_w:
                write("%d:"%w)
                best_w = w
        else:
            L1 += op # undo


def random_distance(code):

    Lx = code.Lx
    m, n = Lx.shape

    best_w = n

    L = Lx[0]

    ops = [Lx[i] for i in range(m)]
        #+ [code.Hx[i] for i in range(code.Hx.shape[0])]

    L = L.copy()
    L1 = L.copy()

    count = 0
    while 1:

        op = choice(ops)
        L1 += op
        L1 %= 2
        w = L1.sum()
        if 0 < w <= best_w:
            L, L1 = L1, L
            if w < best_w:
                write("%d:"%w)
                best_w = w
        else:
            L1 += op # undo

            count += 1
            if count > 1000:
                L = choice(ops)
                L = L.copy()
                L1 = L.copy()
                best_w = n
                count = 0
                write("\n")


def pair_distance(code):

    Lx = code.Lx
    m, n = Lx.shape

    ops = [Lx[i] for i in range(m)]

    best_w = min([op.sum() for op in ops])
    write("%d:"%best_w)

    for i in range(m):
      for j in range(m):
        if i==j:
            continue
        op = (ops[i]+ops[j])%2
        w = op.sum()
        if w < best_w:
            write("%d:"%w)
            best_w = w

    return best_w


def free_distance(code):

    n = code.n

    Hz, Tx = code.Hz, code.Tx

    best_w = n

    M0 = argv.get("M0", 1000)

    for i in range(M0):

        u = zeros2(n)
    
        for j in range(randint(1, 4)):
            idx = randint(0, n-1)
            u[idx] = 1
    
        v = dot2(Hz, u)
    
        T = zeros2(n)
    
        for i in numpy.where(v)[0]:
            T += Tx[i]
            T %= 2
    
        assert dot2(Hz, (u+T)).sum() == 0

        op = (u+T)%2

        if op.sum() and not code.is_stab(op):
    
            w = op.sum()
            if w < best_w:
                print(w, end=' ')
                best_w = w

    #print "OK"
    print()

    return best_w


def lookup_distance(code):
    n = code.n
    Hz, Tx = code.Hz, code.Tx

    d = n
    u = zeros2(n)
    for i0 in range(n):
        u[i0] = 1
        if dot2(Hz, u).sum() == 0:
            #print("*", shortstr(u))
            d = min(d, 1)
        for i1 in range(i0+1, n):
            u[i1] = 1
            if dot2(Hz, u).sum() == 0:
                #print("*", shortstr(u))
                if d>2:
                    print("d=", d)
                d = min(d, 2)
            for i2 in range(i1+1, n):
                u[i2] = 1
                if dot2(Hz, u).sum() == 0:
                    #print("*", shortstr(u))
                    if d>3:
                        print("d=", d)
                    d = min(d, 3)
                for i3 in range(i2+1, n):
                    u[i3] = 1
                    if dot2(Hz, u).sum() == 0:
                        #print("*", shortstr(u))
                        d = min(d, 4)
                    u[i3] = 0
                u[i2] = 0
            u[i1] = 0
        u[i0] = 0
    return d


# is this any faster than the above?
def lookup_distance(code):
    n = code.n
    Hz, Tx = code.Hz, code.Tx

    d = n
    v = zeros2(len(Hz))
    for i0 in range(n):
        v += Hz[:, i0]; v %= 2
        if v.sum() == 0:
            if d>1:
                print("d=1")
            d = min(d, 1)
            return d
        for i1 in range(i0+1, n):
            v += Hz[:, i1]; v %= 2
            if v.sum() == 0:
                if d>2:
                    print("d=2")
                d = min(d, 2)
            for i2 in range(i1+1, n):
                v += Hz[:, i2]; v %= 2
                if v.sum() == 0:
                    if d>3:
                        print("d=3")
                    d = min(d, 3)
                for i3 in range(i2+1, n):
                    v += Hz[:, i3]; v %= 2
                    if v.sum() == 0:
                        if d>4:
                            print("d=4")
                        d = min(d, 4)
                    v += Hz[:, i3] #; v %= 2
                v += Hz[:, i2] #; v %= 2
            v += Hz[:, i1] #; v %= 2
        v += Hz[:, i0] #; v %= 2
        #print(i0, end=" ", flush=True)
    return d


def enum2(n):
    shape = (2,)*n
    for i in range(2**n):
        yield numpy.unravel_index(i, shape)

assert list(enum2(3)) == [
    (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), 
    (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

def enumshape2(m, n):
    for x in enum2(m*n):
        a = array2(x)
        a.shape = m, n
        yield a

#print list(enumshape(2,3))


def stabweights(U, L, verbose=False):

    U = U.copy()
    L = L.copy()

    # Shorten stabilizers -------------

    n = U.shape[0]
    m = n//2
    Hz, Tz = U[:m], U[m:]
    Tx, Hx = L[:m], L[m:]

#    print shortstrx(Hx, Hz)

    #Lx, Lz = zeros2(0, n), zeros2(0, n)
    #code = CSSCode(Lx, Lz, Hx, Tz, Hz, Tx, build=False)
    #print code
    #print code.weightstr()

    ws = []
    for i in range(m):
        Hz1 = pop2(Hz, i)
        graph = Tanner(Hz1)
        #write("i=%d w=%d "%(i, Lz[i].sum()))
        op = graph.localmin(Hz[i], verbose=verbose)
        if op.sum() < Hz[i].sum():
            Hz[i] = op
            write('.')
        else:
            write(choice('\/'))
        ws.append(op.sum())

    for i in range(m):
        Hx1 = pop2(Hx, i)
        graph = Tanner(Hx1)
        #write("i=%d w=%d "%(i, Lx[i].sum()))
        op = graph.localmin(Hx[i], verbose=verbose)
        if op.sum() < Hx[i].sum():
            Hx[i] = op
            write('.')
        else:
            write(choice('\/'))
        ws.append(op.sum())

#    print ws
#    print shortstrx(Hx, Hz)

    return ws

    code = CSSCode(Lx, Lz, Hx, None, Hz, None, build=False)
    print(code)
    print(code.weightstr())

    return code

    # ----------------------------------


def hpack(n, j=4, k=1, check=True, verbose=False):

    write("hpack...")

    U = identity2(n) # upper triangular
    L = identity2(n) # lower triangular
    I = identity2(n)

    for count in range(j*n):

        i = randint(0, n-2)
        j = randint(i+1, n-1)
        #j = randint(0, n-2)
        #if j>=i:
        #    j += 1

        U[i] = (U[i] + U[j])%2
        L[j] = (L[j] - L[i])%2

    assert solve.rank(U) == n
    assert solve.rank(L) == n

    assert eq2(dot2(U, L.transpose()), I)

    if verbose:
        print()
        print(shortstrx(U, L))
        print()

    ws = [n] * n
    ws = stabweights(U, L)

    for i in range(n):
        w = min(U[i].sum(), L[i].sum())
        ws[i] = min(ws[i], w)
    ws = list(enumerate(ws))
    ws.sort(key = lambda item : -item[1])

    idxs = [ws[i][0] for i in range(k)]
    idxs.sort()

    Lx, Lz = zeros2(0, n), zeros2(0, n)

    for idx in reversed(idxs):

        Lz = append2(Lz, U[idx:idx+1])
        Lx = append2(Lx, L[idx:idx+1])
        U = pop2(U, idx)
        L = pop2(L, idx)

    m = (n-k)//2
    Hz, Tz = U[:m], U[m:]
    Tx, Hx = L[:m], L[m:]

    if verbose:
        print()
        print(shortstrx(Hx, Hz))

    write("done.\n")

    code = CSSCode(Lx, Lz, Hx, Tz, Hz, Tx, check=check, verbose=verbose)

    return code


def hpack(n, j=4, k=1, check=True, verbose=False):

    write("hpack...")

    U = identity2(n) # upper triangular
    L = identity2(n) # lower triangular
    I = identity2(n)

    for count in range(j*n):

        i = randint(0, n-1)
        j = randint(0, n-1)
        if i==j:
            continue

        U[i] = (U[i] + U[j])%2
        L[j] = (L[j] - L[i])%2

    assert solve.rank(U) == n
    assert solve.rank(L) == n

    assert eq2(dot2(U, L.transpose()), I)

    if verbose:
        print()
        print(shortstrx(U, L))
        print()

    ws = [n] * n
    ws = stabweights(U, L)

    for i in range(n):
        w = min(U[i].sum(), L[i].sum())
        ws[i] = min(ws[i], w)
    ws = list(enumerate(ws))
    ws.sort(key = lambda item : -item[1])

    idxs = [ws[i][0] for i in range(k)]
    idxs.sort()

    Lx, Lz = zeros2(0, n), zeros2(0, n)

    for idx in reversed(idxs):

        Lz = append2(Lz, U[idx:idx+1])
        Lx = append2(Lx, L[idx:idx+1])
        U = pop2(U, idx)
        L = pop2(L, idx)

    m = (n-k)//2
    Hz, Tz = U[:m], U[m:]
    Tx, Hx = L[:m], L[m:]

    if verbose:
        print()
        print(shortstrx(Hx, Hz))

    write("done.\n")

    code = CSSCode(Lx, Lz, Hx, Tz, Hz, Tx, check=check, verbose=verbose)

    return code


def randcss(n, mx, mz, distance=None, **kw):
    """
    http://arxiv.org/abs/quant-ph/9512032
    Quantum error-correcting codes exist
    with asymptotic rate:
    k/n = 1 - 2H(2t/n) where 
    H(p) = -p log p - (1-p) log (1-p) and
    t = floor((d-1)/2).
    """

    while 1:
        k = n-mx-mz
        assert k>=0
    
        #print "rate:", 1.*k//n
        #H = lambda p: -p*log(p) - (1-p)*log(1-p)
        #d = 56 
        #print 1-2*H(1.*d//n) # works!
    
        Hz = rand2(mz, n)
    
        #print shortstrx(Hx)
    
        kern = numpy.array(solve.find_kernel(Hz))
    
        Hx = zeros2(mx, n)
        for i in range(mx):
            v = rand2(1, n-mx)
            Hx[i] = dot2(v, kern)
        C = CSSCode(Hx=Hx, Hz=Hz, **kw)

        if distance is None:
            break
        d = lookup_distance(C)
        if d < distance:
            continue
        d = lookup_distance(C.dual())
        if d < distance:
            continue
        break

    return C


def sparsecss_FAIL(n, mx, mz, weight=3, **kw):

    print("sparsecss", n, mx, mz)
    k = n-mx-mz
    assert k>=0

    Hz = rand2(mz, n, weight=weight)

    #print shortstrx(Hx)

    kern = numpy.array(solve.find_kernel(Hz))
    mkern = kern.shape[0]
    print("kern:")
    print(shortstr(kern))
    print()


    kern1 = zeros2(mkern, n)
    for i in range(mkern):
        v = rand2(1, mkern)
        kern1[i] = dot2(v, kern)
    print("kern1:")
    print(shortstr(kern1))
    print()

    kern = kern1

    Hx = []
    for i in range(mx):
        j = randint(0, mkern-1)
        v = kern[j].copy()

        count = 0
        while 1:

            v += kern[randint(0, mkern-1)]
            v %= 2

            w = v.sum()

            if w==weight and count > 100:
                break

            count += 1

        Hx.append(v)
    Hx = array2(Hx)
    print(shortstrx(Hx))

    C = CSSCode(Hx=Hx, Hz=Hz, **kw)
    return C


def sparsecss_SLOWSLOW(n, mx, mz, weight=3, **kw):

    print("sparsecss", n, mx, mz)
    k = n-mx-mz
    assert k>=0

    Hz = rand2(mz, n, weight=weight)
    Hx = rand2(mx, n, weight=weight)

    while 1:

        A = dot2(Hz, Hx.transpose())
        A %= 2

        rows = list(range(mz))
        cols = list(range(mx))

        if not A.max():
            break

        #print A.sum(),

        while A.max() and rows and cols:

            i = choice(rows)
            j = choice(cols)

            v = rand2(1, n, weight=weight)

            if random()<=0.5:

                A[i, :] = 0
                Hz[i] = v

                rows.remove(i)

            else:

                A[:, j] = 0
                Hx[j] = v

                cols.remove(j)

    C = CSSCode(Hx=Hx, Hz=Hz, **kw)
    return C


def sparsecss(n, mx, mz, weight=8, **kw):

    print("sparsecss", n, mx, mz)
    k = n-mx-mz
    assert k>=0

    vec = lambda n=n, weight=weight : rand2(1, n, weight=weight)

    Hz = zeros2(0, n)
    Hx = zeros2(0, n)

    Hx = append2(Hx, vec())

    while len(Hz)<mz or len(Hx)<mx:

        # append2 Hz
        rows = shortstr(Hz).split()
        #print rows
        while Hz.shape[0]<mz:
            v = vec()
            u = dot2(Hx, v.transpose())
            if u.sum() == 0 and shortstr(v) not in rows:
                Hz = append2(Hz, v)
                break
            
        # append2 Hx
        rows = shortstr(Hx).split()
        #print rows
        while Hx.shape[0]<mx:
            v = vec()
            u = dot2(Hz, v.transpose())
            if u.sum() == 0 and shortstr(v) not in rows:
                Hx = append2(Hx, v)
                break

        print(shortstrx(Hz, Hx))
        print()

    bits = []
    for i in range(n):
        if Hx[:, i].sum() == 0:
            bits.append(i)
        elif Hz[:, i].sum() == 0:
            bits.append(i)

    for i in reversed(bits):
        Hx = numpy.concatenate(
            (Hx[:, :i], Hx[:, i+1:]), axis=1)
        Hz = numpy.concatenate(
            (Hz[:, :i], Hz[:, i+1:]), axis=1)

    #print shortstrx(Hx, Hz)

    C = CSSCode(Hx=Hx, Hz=Hz, **kw)
    return C


def randldpc(n, mz, rw, check=True, verbose=True):

    Hz = zeros2(mz, n)

    colweights = [0]*n
    for row in range(mz):
        idxs = list(range(n))
        idxs.sort(key = lambda idx : colweights[idx])
        #print [colweights[idx] for idx in idxs]
        for j in range(rw):
            if j==0 and min(colweights)==0: # heuristic...
                i = choice([
                    i for i, idx in enumerate(idxs)
                    if colweights[idx]==0])
                assert colweights[idxs[i]]==0
            else:
                #i = randint(0, len(idxs)-1)
                i = randint(0, len(idxs)//2)
            idx = idxs.pop(i)
            Hz[row, idx] = 1
            colweights[idx] += 1

    #print
    #print colweights

    Hx = zeros2(0, n)
    #Tz = zeros2(0, n)

    #Lx = solve.find_kernel(Hz)
    #Lx = array2(Lx)

    code = CSSCode(Hz=Hz, Hx=Hx)

    print(code.weightstr())

    return code


def randselfdual(m, n, rw):

    assert rw%2 == 0
    H = solve.rand2(1, n, weight=rw)

    while len(H) < m:

        while 1:
            h = solve.rand2(1, n, weight=rw)
            if dot2(H, h.transpose()).sum() == 0:
                break
        H = numpy.concatenate((H, h))

    print(H)
    idxs = numpy.where(H.sum(0)!=0)[0]
    H = H[:, idxs]
    print(H)
    code = CSSCode(Hz=H, Hx=H)

    return code


def ldpc(m, n, cw, rw):

    scount = 0
    while scount < 100:

        H = zeros2(m, n)

        rows = list(range(m))
        cols = list(range(n))
    
        count = 0
        while rows and count < m*n:
    
            row = choice(rows)
    
            wrow = H[row].sum()
            assert wrow < rw
    
            col = choice(cols)
    
            wcol = H[:, col].sum()
            assert wcol < cw
    
            if H[row, col] == 0:
    
                H[row, col] = 1
    
                if wrow==rw-1:
                    rows.remove(row)
                if wcol==cw-1:
                    cols.remove(col)
            count += 1
    
        if not rows:
            break

        scount += 1

    assert not cols, cols

    return H


def x_split(C0, build=True, **kw):
    "split the x-stabilizers of C0"

    #print "x_split"

    mz, n, mx = C0.mz, C0.n+C0.mx, 2*C0.mx

    #print C0.longstr()

    Hz = zeros2(mz, n)
    Hz[:, :C0.n] = C0.Hz

    Hx = zeros2(mx, n)

    #print "Hz:"
    #print shortstrx(Hz)

    for i in range(C0.mx):

        op = C0.Hx[i]
        #Hx[:C0.mx-1, :C0.n] = pop2(C0.Hx, i)
    
        #print "Hx:"
        #print shortstrx(Hx)
    
        idxs = [j for j in range(C0.n) if op[j]]
    
        idxs0 = idxs[:len(idxs)//2]
        idxs1 = idxs[len(idxs)//2:]
        
        #print idxs, idxs0, idxs1
    
        i0, i1 = 2*i, 2*i+1
        for j in idxs0:
            Hx[i0, j] = 1
        for j in idxs1:
            Hx[i1, j] = 1
        Hx[i0, C0.n+i] = 1
        Hx[i1, C0.n+i] = 1

        #print "Hx:"
        #print shortstrx(Hx)
    
        for j in range(mz):
            if dot2(Hz[j], Hx[i0]):
                assert dot2(Hz[j], Hx[i1]) # brilliant!
                Hz[j, C0.n+i] = 1

    C1 = CSSCode(Hx=Hx, Hz=Hz, build=build, **kw)

    #print C1.weightsummary()

    return C1


def classical_distance(H):
    n = H.shape[1]
    dist = n
    for v in solve.find_kernel(H):
        if 0 < v.sum() < dist:
            dist = v.sum()
    return dist


def make_gallagher(r, n, l, m, distance=0):
    assert r%l == 0
    assert n%m == 0
    assert r*m == n*l
    H = zeros2(r, n)
    H1 = zeros2(r//l, n)
    H11 = identity2(r//l)
    #print(H1)
    #print(H11)
    for i in range(m):
        H1[:, (n//m)*i:(n//m)*(i+1)] = H11
    #print(shortstrx(H1))

    while 1:
        H2 = H1.copy()
        idxs = list(range(n))
        for i in range(l):
            H[(r//l)*i:(r//l)*(i+1), :] = H2
            shuffle(idxs)
            H2 = H2[:, idxs]
        Hli = solve.linear_independent(H)
        assert n <= 24, "ummm, too big?"
        dist = classical_distance(Hli)
        if dist >= distance:
            break
    return Hli


def make_gallagher_6(r, n, l, m, distance=0):
    fail = True
    while fail:

        H = make_gallagher(r, n, l, m, distance)

        fail = False
        for i in range(m):
          for j in range(i+1, m):
            v = tuple(H[i] + H[j])
            if v.count(2) > 1:
                fail = True
                break
          if fail:
            break

    return H


def hypergraph_product(H1, H2):
    #print(H1.shape, H2.shape)
    r1, n1 = H1.shape
    r2, n2 = H2.shape
    E1 = identity2(r1)
    E2 = identity2(r2)
    M1 = identity2(n1)
    M2 = identity2(n2)
    Hx = numpy.kron(E2, H1), numpy.kron(H2, E1)
    Hx = numpy.concatenate(Hx, axis=1)
    Hz = numpy.kron(H2.transpose(), M1), numpy.kron(M2, H1.transpose())
    Hz = numpy.concatenate(Hz, axis=1)
    #print(Hx.shape, Hz.shape)
    return Hx, Hz


def todot(H):
    f = open("H.dot", "w")
    m, n = H.shape
    print("graph the_graph\n{", file=f)
    for i in range(m):
        print("  A_%d [shape=box];" % i, file=f)
    for j in range(n):
        print("  L_%d [shape=ellipse];" % j, file=f)
    for i in range(m):
      for j in range(n):
        if H[i, j] == 0:
            continue
        print("  A_%d -- L_%d;" % (i, j), file=f)
    print("}", file=f)
    f.close()



def main():

    m = argv.get('m', 6) # constraints (rows)
    n = argv.get('n', 16) # bits (cols)
    j = argv.get('j', 3)  # column weight (left degree)
    k = argv.get('k', 8)  # row weight (constraint weight; right degree)

    #assert 2*m<=n, "um?"

    max_iter = argv.get('max_iter', 200)
    verbose = argv.verbose
    check = argv.get('check', True)

    strop = shortstr

    Lx = None
    Lz = None
    Hx = None
    Tz = None
    Hz = None
    Tx = None
    build = argv.get('build', True)

    code = None

    if argv.code == 'ldpc':

        H = ldpc(m, n, j, k)
        code = CSSCode(Hz=H, Hx=zeros2(0, n))

    elif argv.code == 'cycle':

        #n = argv.get('n', 20)
        #m = argv.get('m', 18)

        row = str(argv.get('H', '1101'))

        H = []

        for i in range(n):
            h = [0]*n
            for j, c in enumerate(row):
                h[(i+j)%n] = int(c)
            H.append(h)

        delta = n-m
        for i in range(delta):
            H.pop(i*n//delta-i)

#        while len(H) > m:
#            idx = randint(0, len(H)-1)
#            H.pop(idx)

        Hz = array2(H)

        print(shortstr(Hz))

        #print shortstr(solve.row_reduce(Hz))

#        code = CSSCode(Hz=Hz)
#        print
#        print code
#        print code.weightstr()
#
#        decoder = StarDynamicDistance(code)
#        L = decoder.find(verbose=verbose)
#        print shortstr(L)
#        print "distance:", L.sum()
#        return

    elif argv.code == "opticycle":

        code = opticycle(m, n)

        H = Hz = code.Hz

        #return

    elif argv.code == 'tree':

        k = argv.get('k', 5)
        n = 2**k + 1

        code = build_tree(45)
        print(code)

#        graph = Graph()
#        for i in range(H.shape[0]):
#            edge = list(numpy.where(H[i])[0])
#            #print edge,
#            for i0 in edge:
#              for i1 in edge:
#                if i0!=i1:
#                    graph.add_edge(i0, i1)
#
#        leaves = [idx for idx in range(n) if H[:, idx].sum()<=1]
#        print leaves
#
#
#        k = len(leaves)//2
#        print graph.metric(leaves[1], leaves[k])
#        return
#        for i in range(k-1):
#            #i0 = poprand(leaves)
#            #i1 = poprand(leaves)
#            u = zeros2(1, n)
#            u[0, leaves[i]] = 1
#            u[0, leaves[i+k]] = 1
#            # what about picking three leaves?
#            print (leaves[i], leaves[i+k]),
#            H = append2(H, u)
#        print
#
#        # if we take all the leaves we get degenerate...
#        leaves = [idx for idx in range(n) if H[:, idx].sum()<=1]
#        print "leaves:", leaves
#
#        #print shortstr(H)
#        #print
#        #print shortstr(solve.row_reduce(H))

#        code = CSSCode(Hz=H)
#        print code
#        print code.weightstr()

#        print numpy.where(code.Lx[3])
#
#        return

    elif argv.code == 'bicycle':

        from qupy.ldpc.bicycle import Bicycle_LDPC
        b = Bicycle_LDPC(m, n, j, k)
        Hx = Hz = b.H

    elif argv.code == 'toric':

        from qupy.ldpc.toric import Toric2D
        l = argv.get('l', 8)

        toric = Toric2D(l)
        Hx, Hz = toric.Hx, toric.Hz
        strop = toric.strop
        #print("Hx:")
        #print(shortstr(Hx))
        #print("Lx:")
        #print(shortstr(toric.Lx))

    elif argv.code == 'torichie':

        from qupy.ldpc.toric import Toric2DHie
        l = argv.get('l', 8)

        toric = Toric2DHie(l)
        Hx, Hz = toric.Hx, toric.Hz

    elif argv.code == 'surface':

        from qupy.ldpc.toric import Surface
        l = argv.get('l', 8)

        surface = Surface(l)
        #Hx, Hz = surface.Hx, surface.Hz
        #strop = surface.strop
        #print shortstr(Hx)
        code = surface.get_code()

    elif argv.code == 'toric3':

        from qupy.ldpc.toric import Toric3D
        l = argv.get('l', 8)

        toric = Toric3D(l)
        Hx, Hz = toric.Hx, toric.Hz
        strop = toric.strop
        #print shortstr(Hx)

    elif argv.code == 'gcolor':

        from qupy.ldpc.gcolor import  Lattice
        l = argv.get('l', 1)
        lattice = Lattice(l)
        code = lattice.build_code()

        #print "Gx:", code.Gx.shape
        #print shortstr(code.Gx)
        print("Gx rank:", solve.rank(code.Gx))
        print("Hx:")
        print(shortstr(code.Hx))

    elif argv.code == 'self_ldpc':

        from qupy.ldpc.search import SelfContainingLDPC_Code
        ldpc_code = SelfContainingLDPC_Code(m, n, j, k)
        ldpc_code.search(max_step=argv.get('max_step', None), verbose=verbose)
        #print ldpc_code

        H = ldpc_code.H
        print("H:")
        print(shortstr(H))
        H = solve.linear_independent(H)
        Hx = Hz = H

        code = CSSCode(Hx=Hx, Hz=Hz)

    elif argv.code == 'cssldpc':

        from qupy.ldpc.search import CSS_LDPC_Code
        ldpc_code = CSS_LDPC_Code(m, n, j, k)
        ldpc_code.search(max_step=argv.get('max_step', None), verbose=verbose)
        #print ldpc_code

        Hx, Hz = ldpc_code.Hx, ldpc_code.Hz
        #print "Hx:"
        #print shortstr(Hx)
        Hx = solve.linear_independent(Hx)
        Hz = solve.linear_independent(Hz)
        #Hx = Hz = H

        code = CSSCode(Hx=Hx, Hz=Hz)

    elif argv.code == "hpack":

        code = hpack(n, j=j, k=k, check=check, verbose=verbose)

    elif argv.code == 'randldpc':

        mz = argv.get('mz', n//3)
        rw = argv.get('rw', 12) # row weight
        code = randldpc(n, mz, rw, check=check, verbose=verbose)

    elif argv.code == 'randcss':

        mx = argv.get('mx', n//3)
        mz = argv.get('mz', n//3)
        distance = argv.get("code_distance")
        code = randcss(n, mx, mz, distance=distance, 
            check=check, verbose=verbose)

    elif argv.code == 'sparsecss':

        mx = argv.get('mx', n//3)
        mz = argv.get('mz', n//3)
        weight = argv.get('weight', 4)
        code = sparsecss(n, mx, mz, weight, check=check, verbose=verbose)

    elif argv.code == 'ensemble':

        mx = argv.get('mx', n//3)
        mz = argv.get('mz', mx)
        rw = argv.get('rw', 12) # row weight
        maxw = argv.get('maxw', 4*rw)
        C = argv.get("C", 100)
        code = ensemble.build(n, mx, mz, rw, maxw, C,
            check=check, verbose=verbose)

    elif argv.code == "gallagher":
        r = argv.get("r", 6) # rows
        n = argv.get("n", 8) # cols
        l = argv.get("l", 3) # column weight
        m = argv.get("m", 4) # row weight
        distance = argv.get("distance", 4)
        H1 = make_gallagher_6(r, n, l, m, distance)
        print(shortstr(H1))
        dist = classical_distance(H1)
        print("dist:", dist)
        H2 = H1.transpose()
        Hx, Hz = hypergraph_product(H1, H2)
        code = CSSCode(Hx=Hx, Hz=Hz)

    elif argv.code == 'hamming':
        H1 = parse("""
        ...1111
        .11..11
        1.1.1.1
        """)
        dist = classical_distance(H1)
        print("dist:", dist)
        H2 = H1.transpose()
        Hx, Hz = hypergraph_product(H1, H2)
        code = CSSCode(Hx=Hx, Hz=Hz)
        #print(code)
        #return

    elif argv.code == "joschka12":
        # [12, 3, 6]
        H1 = numpy.array(
            [[1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
            [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0]])
        dist = classical_distance(H1)
        print("dist:", dist)
        H2 = H1.transpose()
        Hx, Hz = hypergraph_product(H1, H2)
        # [[255, 9, 6]]
        code = CSSCode(Hx=Hx, Hz=Hz)

    elif argv.code == "joschka16":
        # [16, 4, 6]
        H1 = numpy.array([[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
       [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
       [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
       [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
       [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
       [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
       [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]])
        dist = classical_distance(H1)
        print("dist:", dist)
        H2 = H1.transpose()
        Hx, Hz = hypergraph_product(H1, H2)
        # [[400, 16, 6]]
        code = CSSCode(Hx=Hx, Hz=Hz)

    elif argv.code == "joschka20":
        # [20, 5, 8]
        H1 = numpy.array(
        [[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
       [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
       [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
       [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
       [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
       [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]])
        dist = classical_distance(H1)
        print("dist:", dist)
        H2 = H1.transpose()
        Hx, Hz = hypergraph_product(H1, H2)
        # [[625, 25, 8]]
        code = CSSCode(Hx=Hx, Hz=Hz)

    elif argv.code == "joschka24":
        # [24, 6, 10]
        H1 = numpy.array(
        [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
       [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
       [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
       [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
       [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
        dist = classical_distance(H1)
        print("dist:", dist)
        H2 = H1.transpose()
        Hx, Hz = hypergraph_product(H1, H2)
        # [[900, 36, 10]]
        code = CSSCode(Hx=Hx, Hz=Hz)

    elif argv.code == "joschka28":
        # [28, 7, 10]
        H1 = numpy.array(
        [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 1, 0, 0, 0],
       [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 1, 0, 0],
       [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 1, 0, 0],
       [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0],
       [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1,
        0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 1, 0, 1],
       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0],
       [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1]])
        dist = classical_distance(H1)
        print("dist:", dist)
        H2 = H1.transpose()
        Hx, Hz = hypergraph_product(H1, H2)
        # [[1225, 49, 10]]
        code = CSSCode(Hx=Hx, Hz=Hz)

    elif argv.code == "joschka32":
        # [32, 8, 10]
        H1 = numpy.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
       [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 1, 0, 0, 0]])
        dist = classical_distance(H1)
        print("dist:", dist)
        H2 = H1.transpose()
        Hx, Hz = hypergraph_product(H1, H2)
        # [[1600, 64, 10]]
        code = CSSCode(Hx=Hx, Hz=Hz)

    elif argv.code == "joschka36":
        # [36, 9, 12]
        H1 = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
       [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        dist = classical_distance(H1)
        print("dist:", dist)
        H2 = H1.transpose()
        Hx, Hz = hypergraph_product(H1, H2)
        # [[, 81, 12]]
        code = CSSCode(Hx=Hx, Hz=Hz)

    elif argv.code == "joschka40":
        H1 = numpy.array(
    [[0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0],
    [0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]])
    
        dist = classical_distance(H1)
        print("dist:", dist)
        H2 = H1.transpose()
        Hx, Hz = hypergraph_product(H1, H2)
        code = CSSCode(Hx=Hx, Hz=Hz)

    elif argv.code == "joschka44":
        H1 = numpy.array(
    [[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0],
    [0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    [1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0]])
        dist = classical_distance(H1)
        print("dist:", dist)
        H2 = H1.transpose()
        Hx, Hz = hypergraph_product(H1, H2)
        code = CSSCode(Hx=Hx, Hz=Hz)

    elif argv.code == "joschka60":
        H1 = numpy.array(
    [[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
    0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,
    0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,
    1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,
    0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,
    0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
        dist = classical_distance(H1)
        print("dist:", dist)
        H2 = H1.transpose()
        Hx, Hz = hypergraph_product(H1, H2)
        code = CSSCode(Hx=Hx, Hz=Hz)

    elif argv.code == "qr7":
        H = parse("""
        ...1111
        .11..11
        1.1.1.1
        """)
        code = CSSCode(Hx=H, Hz=H)

    elif argv.code == "golay" or argv.code == "qr23":
        H = parse("""
        1.1..1..11111..........
        1111.11.1....1.........
        .1111.11.1....1........
        ..1111.11.1....1.......
        ...1111.11.1....1......
        1.1.1.111..1.....1.....
        1111...1..11......1....
        11.111...11........1...
        .11.111...11........1..
        1..1..11111..........1.
        .1..1..11111..........1
        """)
        code = CSSCode(Hx=H, Hz=H)

    elif argv.code == "qr31":
        H = parse("""
        1..1..1.1...11.11..............
        11.11.1111..1.11.1.............
        11111111.11.1.....1............
        .11111111.11.1.....1...........
        ..11111111.11.1.....1..........
        ...11111111.11.1.....1.........
        1..111.1.1111.11......1........
        11.111....11...........1.......
        .11.111....11...........1......
        ..11.111....11...........1.....
        ...11.111....11...........1....
        ....11.111....11...........1...
        1..1.1...11.11..............1..
        .1..1.1...11.11..............1.
        ..1..1.1...11.11..............1
        """)
        code = CSSCode(Hx=H, Hz=H)

    elif argv.code == "qr17":
        # not self-dual
        Hz = parse("""
        .11.1...11...1.11
        1.11.1...11...1.1
        11.11.1...11...1.
        .11.11.1...11...1
        1.11.11.1...11...
        .1.11.11.1...11..
        ..1.11.11.1...11.
        ...1.11.11.1...11
        1...1.11.11.1...1
        11...1.11.11.1...
        .11...1.11.11.1..
        ..11...1.11.11.1.
        ...11...1.11.11.1
        1...11...1.11.11.
        .1...11...1.11.11
        1.1...11...1.11.1
        11.1...11...1.11.
        """)
        Hx = parse("""
        ...1.111..111.1..
        11.1.....1.111..1
        111..111.1.....1.
        ....1.111..111.1.
        111.1.....1.111..
        .111..111.1.....1
        .....1.111..111.1
        .111.1.....1.111.
        1.111..111.1.....
        1.....1.111..111.
        ..111.1.....1.111
        .1.111..111.1....
        .1.....1.111..111
        1..111.1.....1.11
        ..1.111..111.1...
        1.1.....1.111..11
        11..111.1.....1.1
        """)
        Hz = solve.linear_independent(Hz)
        Hx = solve.linear_independent(Hx)
        code = CSSCode(Hx=Hx, Hz=Hz)

    elif argv.code == "qr41":
        # not self-dual
        Hz = parse("""
        .11.11..111.....1.1.11.1.1.....111..11.11
        1.11.11..111.....1.1.11.1.1.....111..11.1
        11.11.11..111.....1.1.11.1.1.....111..11.
        .11.11.11..111.....1.1.11.1.1.....111..11
        1.11.11.11..111.....1.1.11.1.1.....111..1
        11.11.11.11..111.....1.1.11.1.1.....111..
        .11.11.11.11..111.....1.1.11.1.1.....111.
        ..11.11.11.11..111.....1.1.11.1.1.....111
        1..11.11.11.11..111.....1.1.11.1.1.....11
        11..11.11.11.11..111.....1.1.11.1.1.....1
        111..11.11.11.11..111.....1.1.11.1.1.....
        .111..11.11.11.11..111.....1.1.11.1.1....
        ..111..11.11.11.11..111.....1.1.11.1.1...
        ...111..11.11.11.11..111.....1.1.11.1.1..
        ....111..11.11.11.11..111.....1.1.11.1.1.
        .....111..11.11.11.11..111.....1.1.11.1.1
        1.....111..11.11.11.11..111.....1.1.11.1.
        .1.....111..11.11.11.11..111.....1.1.11.1
        1.1.....111..11.11.11.11..111.....1.1.11.
        .1.1.....111..11.11.11.11..111.....1.1.11
        """)
        Hx = parse("""
        ...1..11...11111.1.1..1.1.11111...11..1..
        1111...11..1.....1..11...11111.1.1..1.1.1
        111.1.1..1.1.11111...11..1.....1..11...11
        ....1..11...11111.1.1..1.1.11111...11..1.
        11111...11..1.....1..11...11111.1.1..1.1.
        1111.1.1..1.1.11111...11..1.....1..11...1
        .....1..11...11111.1.1..1.1.11111...11..1
        .11111...11..1.....1..11...11111.1.1..1.1
        11111.1.1..1.1.11111...11..1.....1..11...
        1.....1..11...11111.1.1..1.1.11111...11..
        1.11111...11..1.....1..11...11111.1.1..1.
        .11111.1.1..1.1.11111...11..1.....1..11..
        .1.....1..11...11111.1.1..1.1.11111...11.
        .1.11111...11..1.....1..11...11111.1.1..1
        ..11111.1.1..1.1.11111...11..1.....1..11.
        ..1.....1..11...11111.1.1..1.1.11111...11
        1.1.11111...11..1.....1..11...11111.1.1..
        ...11111.1.1..1.1.11111...11..1.....1..11
        1..1.....1..11...11111.1.1..1.1.11111...1
        .1.1.11111...11..1.....1..11...11111.1.1.
        """)
        code = CSSCode(Hz=Hz, Hx=Hx)

    elif argv.code == "qr47":
        H = parse("""
        1...11..11.11..1..1.1..11......................
        11..1.1.1.11.1.11.1111.1.1.....................
        111.1..11.....111111.111..1....................
        11111......11...11.1..1....1...................
        .11111......11...11.1..1....1..................
        1.11..1.11.11111...111.1.....1.................
        11.1.1.11.11.11.1.1..111......1................
        111..11.......1..1111.1........1...............
        .111..11.......1..1111.1........1..............
        1.11.1.1.1.11..11.11.111.........1.............
        11.1.11..111.1.11111..1...........1............
        .11.1.11..111.1.11111..1...........1...........
        1.111..1.1...1...1.1.1.1............1..........
        11.1.....1111.11......11.............1.........
        111..1..111..1..1.1.1.................1........
        .111..1..111..1..1.1.1.................1.......
        ..111..1..111..1..1.1.1.................1......
        ...111..1..111..1..1.1.1.................1.....
        1.....1.1..1.111.11...11..................1....
        11..11.11..1..1.1..11......................1...
        .11..11.11..1..1.1..11......................1..
        ..11..11.11..1..1.1..11......................1.
        ...11..11.11..1..1.1..11......................1
        """)
        code = CSSCode(Hx=H, Hz=H)

    elif argv.code == "qr71":
        H = parse("""
        1.1.1.11.1...11..11.....1....1...1111..................................
        1111111.111..1.1.1.1....11...11..1...1.................................
        .1111111.111..1.1.1.1....11...11..1...1................................
        ..1111111.111..1.1.1.1....11...11..1...1...............................
        1.11.1..1..11.1.11..1.1.1..111..1.11....1..............................
        1111...1....1.11.....1.111..1.1...1......1.............................
        .1111...1....1.11.....1.111..1.1...1......1............................
        1..1.111.....1..1.1....11111.11.1111.......1...........................
        111.....11...1....11.....1111111............1..........................
        .111.....11...1....11.....1111111............1.........................
        ..111.....11...1....11.....1111111............1........................
        ...111.....11...1....11.....1111111............1.......................
        ....111.....11...1....11.....1111111............1......................
        1.1.11...1.......1.....1.....1111................1.....................
        .1.1.11...1.......1.....1.....1111................1....................
        ..1.1.11...1.......1.....1.....1111................1...................
        ...1.1.11...1.......1.....1.....1111................1..................
        1.1....11.....1..11..1..1..1.1.......................1.................
        .1.1....11.....1..11..1..1..1.1.......................1................
        ..1.1....11.....1..11..1..1..1.1.......................1...............
        ...1.1....11.....1..11..1..1..1.1.......................1..............
        ....1.1....11.....1..11..1..1..1.1.......................1.............
        .....1.1....11.....1..11..1..1..1.1.......................1............
        ......1.1....11.....1..11..1..1..1.1.......................1...........
        1.1.1.1......1.1.11..1...1..11.1.1.1........................1..........
        1111111..1...1..11.1..1.1.1...1.11.1.........................1.........
        11.1.1...11..1......1..111.1.1.1...1..........................1........
        11.....1.111.1...11..1...11.111.1111...........................1.......
        11..1.11111111...1.1..1.1.11..11................................1......
        .11..1.11111111...1.1..1.1.11..11................................1.....
        ..11..1.11111111...1.1..1.1.11..11................................1....
        ...11..1.11111111...1.1..1.1.11..11................................1...
        ....11..1.11111111...1.1..1.1.11..11................................1..
        1.1.11.1...11..11.....1....1...1111..................................1.
        .1.1.11.1...11..11.....1....1...1111..................................1
        """)
        code = CSSCode(Hx=H, Hz=H)

    elif argv.code == "rm_2_5_p":
        H = parse("""
        ...11111111....................
        .11..1111..11..................
        1.1.1.11.1.1.1.................
        11.1..1.11.1..1................
        ...1111........1111............
        .11..11........11..11..........
        1.1.1.1........1.1.1.1.........
        11.1..1.........11.1..1........
        .11111111......11......11......
        1.111111.1.....1.1.....1.1.....
        11.1111.11......11.....1..1....
        111.1111...1...1...1...1...1...
        1111.11.1..1....1..1...1....1..
        11111.1..1.1.....1.1...1.....1.
        111111.111.1...111.1...1......1
        """)
        code = CSSCode(Hx=H, Hz=H)

    elif argv.code == "to_16_6": # triorthogonal
        H = parse("""
        ........11111111
        ....1111....1111
        11111111........
        ..11..11..11..11
        .1.1.1.1.1.1.1.1
        1..1.11..11.1..1
        """) # distance = 4
        code = CSSCode(Hx=H, Hz=H)

    elif argv.code == "randselfdual":
        m = argv.get("m", 4)
        n = argv.get("n", 8)
        rw = argv.get("rw", 4)
        code = randselfdual(m, n, rw)

    if argv.truncate:
        idx = argv.truncate
        Hx = Hx[:idx]
        Hz = Hz[:idx]

    for arg in argv:
        if arg.endswith('.ldpc'):
            code = CSSCode.load(arg, build=build, check=check,
                rebuild=argv.rebuild)
    
            if argv.rebuild:
                code.save(arg)

            break

    if code is None:
        code = CSSCode(Lx, Lz, Hx, Tz, Hz, Tx,
            build=build, check=check, verbose=verbose)

    if argv.dual:
        print("dual code...")
        code = code.dual(build=build)

    # take several copies of the code..
    mul = argv.get("mul", 1)
    code = mul * code

    if argv.classical_product:
        H1 = code.Hx
        H2 = H1.transpose()
        Hx, Hz = hypergraph_product(H1, H2)
        code = CSSCode(Hx=Hx, Hz=Hz)

    if argv.product:

        codes = code.product(code.dual())

        for _code in codes:
            print(_code)
            if _code.k:
                _code.save(stem="prod")

    if argv.verbose != 0:
        print(code)
        #print code.longstr()
        #print code.weightstr()
        print(code.weightsummary())

    if argv.shorten:
        code.shorten()
        print(code)
        print(code.weightsummary())

    if argv.prune:
        print("pruning:", argv.prune)
        code.prune_xlogops(argv.prune)
        print(code)
        print(code.weightsummary())

    if argv.split:

        for i in range(4):
            code = x_split(code, build=False)
            print(code)
            print(code.weightsummary())
            code = code.dual()
            code = x_split(code, build=False)
            code = code.dual()
            print(code)
            print(code.weightsummary())
            code.save(stem="split")

    if argv.save:
        code.save(argv.save)

    if argv.symplectic:
        i = randint(0, code.mz-1)
        j = randint(0, code.k-1)
        code.Lz[j] += code.Hz[i]
        code.Lz[j] %= 2
        code.Tx[i] -= code.Lx[j]
        code.Tx[i] %= 2
        code.do_check()
    
        return

    if argv.showcode:
        print(code.longstr())

    if argv.todot:
        todot(code.Hz)
        return

    if argv.tanner:
        Hx = code.Hx
        graph = Tanner(Hx)
        #graph.add_dependent()

        depth = 3

        graphs = [graph]
        for i in range(depth):
            _graphs = []
            for g in graphs:
                left, right = g.split()
                _graphs.append(left)
                _graphs.append(right)
            graphs = _graphs
        print("split", graphs)

        k = code.Lx.shape[0]
        op = code.Lx[1]
        #op = (op + code.Tx[20])%2
        print(strop(op), op.sum())
        print()
        for i in range(Hx.shape[0]):
            if random()<=0.5:
                op = (op+Hx[i])%2
        #print shortstr(op), op.sum()

        print(strop(op), op.sum())
        print()

        #for graph in [left, right]:
        for graph in graphs*2:
        
            #op = graph.minimize(op, verbose=True)
            op = graph.localmin(op, verbose=True)
            print(strop(op), op.sum())
            print()

        #op = graph.localmin(op)
        #print strop(op), op.sum()
        #print
        #print shortstr(op), op.sum()
            
        return

    if argv.distance=='star':
        decoder = StarDynamicDistance(code)
        L = decoder.find(verbose=verbose)
        print(shortstr(L))
        print("distance <=", L.sum())
    elif argv.distance=='stab':
        d = random_distance_stab(code)
        print("distance <=", d)
    elif argv.distance=='pair':
        d = pair_distance(code)
        print("distance <=", d)
    elif argv.distance=='free':
        d = free_distance(code)
        print("distance <=", d)
    elif argv.distance=='lookup':
        d = lookup_distance(code)
        if d <= 4:
            print("distance =", d)
        else:
            print("distance <=", d)

    if type(argv.distance)==str:
        return

    if argv.get('exec'):

        exec(argv.get('exec'))

    N = argv.get('N', 0)
    p = argv.get('p', 0.01)
    weight = argv.weight

    assert code.Tx is not None, "use code.build ?"

    decoder = get_decoder(argv, argv.decode, code)
    if decoder is None:
        return

    if argv.whack:
        whack(code, decoder, p, N, 
            argv.get("C0", 10), argv.get("error_rate", 0.05), argv.get("mC", 1.2),
            verbose, argv)
        return

    if argv.noerr:
        print("redirecting stderr to stderr.out")
        fd = os.open("stderr.out", os.O_CREAT|os.O_WRONLY)
        os.dup2(fd, 2)

    decoder.strop = strop
    print("decoder:", decoder.__class__.__name__)

    n_ranks = numpy.zeros((code.n,), dtype=numpy.float64)
    n_record = []
    k_ranks = numpy.zeros((code.k,), dtype=numpy.float64)
    k_record = []

    failures = open(argv.savefail, 'w') if argv.savefail else None

    if argv.loadfail:
        loadfail = open(argv.loadfail)
        errs = loadfail.readlines()
        N = len(errs)

    newHx = []
    newHz = []

    distance = code.n
    count = 0
    failcount = 0
    nonuniq = 0

    for i in range(N):

        # We use Hz to look at X type errors (bitflip errors)

        if argv.loadfail:
            err_op = parse(errs[i])
            err_op.shape = (code.n,)
        elif weight is not None:
            err_op = zeros2(code.n)
            r = 0
            while r < weight:
                idx = randint(0, code.n-1)
                if err_op[idx] == 0:
                    err_op[idx] = 1
                    r += 1
            #print err_op
        else:
            err_op = ra.binomial(1, p, (code.n,))
            err_op = err_op.astype(numpy.int32)

#        err_op = parse("..............................")
#        err_op.shape = (err_op.shape[1],)
        write(str(err_op.sum()))
        #print err_op.shape

        s = dot2(code.Hz, err_op)
        write(":s%d:"%s.sum())

        op = decoder.decode(p, err_op, verbose=verbose, argv=argv)

        c = 'F'
        success = False
        if op is not None:
            op = (op+err_op)%2
            # Should be a codeword of Hz (kernel of Hz)
            if dot2(code.Hz, op).sum() != 0:
                print(dot2(code.Hz, op))
                print("\n!!!!!  BUGBUG  !!!!!", sparsestr(err_op))
                continue
            write("%d:"%op.sum())

            #print "is_stab:"
            #print strop(op)
            # Are we in the image of Hx ? If so, then success.
            success = code.is_stab(op)

            if success and op.sum():
                nonuniq += 1
            #    print "\n", shortstr(err_op)
            #    return

            c = '.' if success else 'x'

            if argv.k_rank:

                # XX this does not work very well...

                a = dot2(code.Lz, op)
                assert (a.sum()==0) == success

                if not success:
                    #r = -1. // (a.sum()**0.5) # sqrt ??
                    r = -1. / a.sum()
                    k_ranks += r*a
                else:
                    #r = 1. / (code.k**0.5)
                    r = 1. / code.k
                    k_ranks += r*a

                k_record.append((a, success))


            if op.sum() and not success:
                distance = min(distance, op.sum())

                if argv.minop and argv.minop >= op.sum():
                    a = dot2(code.Lz, op)
                    print()
                    print(sparsestr(a))

            if not success and argv.addstabs and op.sum()<=argv.addstabs:
                #write("%d"%op.sum())
                print((sparsestr(op)))
                newHx.append(op)

                A = numpy.concatenate((code.Lx, code.Hx))
                c = solve.solve(A.transpose(), op.transpose()).transpose()
                cL = c[:code.k]
                zstab = dot2(cL, code.Lz)
                print(shortstr(zstab))
                graph = Tanner(code.Hz)
                zstab = graph.minimize(zstab, target=8, verbose=True)
                print(shortstr(zstab))
                newHz.append(zstab)
                break

#                Hz, Hx = code.Hz, append2(code.Hx, op)
#                #if len(Hz) < len(Hx):
#                #    Hx, Hz = Hz, Hx
#                print Hx.shape, Hz.shape
#                code = CSSCode(Hx=Hx, Hz=Hz)
#                if len(Hz) < len(Hx):
#                    code = code.dual()
#                    decoder = get_decoder(argv, argv.decode, code)
#                print
#                print code
#                print code.weightsummary()
#                code.save("addstabs_%d_%d_%d.ldpc"%(code.n, code.k, code.distance))

            if not success and argv.showfail:
                print()
                print("FAIL:")
                print(shortstr(err_op))

        else:
            failcount += 1

            if failures:
                print(shortstr(err_op), file=failures)
                failures.flush()

        if argv.n_rank:
            r = 2.*int(success) - 1
            #print "r =", r
            if err_op.sum():
                r /= err_op.sum()
                #print r
                #print r*err_op
                n_ranks += r*err_op
                n_record.append((err_op, success))

        write(c+' ')
        count += success

    if hasattr(decoder, 'fini'):
        decoder.fini()

    if N:
        print()
        print(datestr)
        print(argv)
        print("error rate = %.8f" % (1. - 1.*count / (i+1)))
        print("fail rate = %.8f" % (1.*failcount / (i+1)))
        print("nonuniq = %d" % nonuniq)
        print("distance <= %d" % distance)

    if argv.n_rank:
        #n_ranks //= N
        n_ranks = list(enumerate(n_ranks))
        n_ranks.sort(key = lambda item : -item[1])
        #print [r[0] for r in n_ranks]
        print(' '.join('%d:%.2f'%r for r in n_ranks))
        #print min(r[1] for r in n_ranks)
        #print max(r[1] for r in n_ranks)
        A = numpy.array([rec[0] for rec in n_record])
        b = numpy.array([rec[1] for rec in n_record])
        print(A.shape, b.shape)
        sol = lstsq(A, b)
        x, res, rk, s = sol
        #print x
        for i, val in enumerate(x):
            if val<0:
                print(i, end=' ')
        print()

    if argv.k_rank:
        #k_ranks //= N
        k_ranks = list(enumerate(k_ranks))
        k_ranks.sort(key = lambda item : -item[1])
        #print [r[0] for r in k_ranks]
        print(' '.join('%d:%.2f'%r for r in k_ranks))
        #print ' '.join(str(r) for r in k_ranks)
        #print min(r[1] for r in k_ranks)
        #print max(r[1] for r in k_ranks)
        A = numpy.array([rec[0] for rec in k_record])
        b = numpy.array([rec[1] for rec in k_record])
        #print A.shape, b.shape
        sol = lstsq(A, b)
        x, res, rk, s = sol
        #print x
        xs = list(enumerate(x))
        xs.sort(key = lambda item : item[1])
        print(' '.join('%d:%.2f'%item for item in xs))

    if argv.addstabs and newHz:
        Hz = append2(code.Hz, newHz[0])
        code = CSSCode(Hx=code.Hx, Hz=Hz)
        code.save(stem="addstabs")
        print(code)
        print(code.weightsummary())

    elif argv.addstabs and newHx: # and newHx.shape[0]>code.Hx.shape[0]:
        newHx = array2(newHx)
        newHx = numpy.concatenate((code.Hx, newHx))
        #print shortstr(newHx)
        #print
        Hx = solve.linear_independent(newHx)
        #print shortstr(Hx)
        #print
        code = CSSCode(Hx=Hx, Hz=code.Hz)
        code.save(stem="addstabs")
        print(code)
        print(code.weightsummary())
    

class MultiDecoder(object):
    def __init__(self, decode0, decode1):
        self.decode0 = decode0
        self.decode1 = decode1

    def decode(self, p, err_op, **kw):
        decode0 = self.decode0
        decode1 = self.decode1
        op = decode0.decode(p, err_op, **kw)
        if op is None:
            write("F")
            op = decode1.decode(p, err_op, **kw)
        return op


def choose(items, n):
    m = len(items)
    if n==0:
        yield []
    elif n==1:
        for item in items:
            yield [item,]
    elif n==2:
        for i in range(m):
          for j in range(i+1, m):
            yield [items[i], items[j]]
    elif n==3:
        for i in range(m):
         for j in range(i+1, m):
          for k in range(j+1, m):
            yield [items[i], items[j], items[k]]
    else:
        for i in range(m):
            for _items in choose(items[i+1:], n-1):
                yield [items[i],]+_items



class LookupDecoder(object):
    """
        Keep a lookup table of syndromes for all possible errors
        up to a given weight. Use this to decode.
        This is a minimum weight decoder, when it succeeds.
    """
    def __init__(self, code, weight=4, maxsize=256*1024):
        lookup = {}
        n = code.n
        Hz = code.Hz
        Lz = code.Lz
        err = zeros2(n)
        write("LookupDecoder: building...")
        w = 0
        #for w in range(0, weight+1):
        count = 0
        remove = set()
        while w < weight+1 or len(lookup)<maxsize:
            for idxs in choose(list(range(n)), w):
                err[:] = 0
                err[idxs] = 1
                syn = dot2(Hz, err)
                key = syn.tostring()
                other = lookup.get(key)
                if other is None:
                    lookup[key] = idxs
                elif len(other)==w:
                    for idx in other:
                        err[idx] = (err[idx]+1)%2
                    assert dot2(Hz, err).sum()==0
                    #assert dot2(Lz, err).sum()==0, "not an exact decoder."
                    if dot2(Lz, err).sum():
                        remove.add(key)
            if len(lookup)==count:
                break
            count = len(lookup)
            w += 1
        write("\n")
        for key in remove:
            del lookup[key]
        print("lookup size:", len(lookup))
        self.lookup = lookup
        self.code = code

    def decode(self, p, err, **kw):
        code = self.code
        Hz = code.Hz
        syn = dot2(Hz, err)
        #if err.sum() <=
        a = syn.tostring()
        idxs = self.lookup.get(a)
        if idxs is None:
            #write("?")
            return None
        op = zeros2(code.n)
        op[idxs] = 1
        return op



def get_decoder(argv, decode, code):

    d = 2

    decoder = None

    if decode is None:
        pass

    elif '+' in decode:

        items = decode.split('+')
        a = items[0]
        b = '+'.join(items[1:])
        a = get_decoder(argv, a, code)
        b = get_decoder(argv, b, code)
        decoder = MultiDecoder(a, b)

    elif decode=='bp':

        #argv.noerr = True # supress output of RadfordBPDecoder

        decoder = RadfordBPDecoder(d, code.Hz)
        #decoder = PyCodeBPDecoder(d, code.Hz)

    elif decode=='random':
        decoder = RandomDecoder(code)

    elif decode=='metro':
        from qupy.ldpc.decoder import MetroDecoder
        decoder = MetroDecoder(code)

    elif decode=='metrologop':
        from qupy.ldpc.decoder import MetroLogopDecoder
        decoder = MetroDecoder(code)

    elif decode=='star':
        from qupy.ldpc.decoder import StarMetroDecoder
        decoder = StarMetroDecoder(code)

    elif decode=='tempered':
        from qupy.ldpc.decoder import TemperedDecoder
        decoder = TemperedDecoder(code)

    elif decode=='cluster':
        decoder = ClusterCSSDecoder(2, code.Hx, code.Hz)

    elif decode=='mincluster':
        decoder = ClusterCSSDecoder(2, code.Hx, code.Hz)
        decoder.minimize = True

    elif decode=='pma':
        decoder = PMADecoder(code)

    elif decode=='dynamic':
        decoder = DynamicDecoder(code)

    elif decode=='stardynamic':
        decoder = StarDynamicDecoder(code)

    elif decode=='simple':
        from qupy.ldpc.decoder import SimpleDecoder
        decoder = SimpleDecoder(code)

    elif decode=='schoning':
        from qupy.ldpc.decoder import SchoningDecoder
        decoder = SchoningDecoder(code)

    elif decode=='ml':
        decoder = MLDecoder(code)

    elif decode=='ensemble':
        C = argv.get("C", 40)
        K = argv.get("K", 1)
        decoder = ensemble.EnsembleDecoder(code, C, K)

    elif decode=='multiensemble':
        C = argv.get("C", 40)
        K = argv.get("K", 5)
        decoder = ensemble.MultiEnsembleDecoder(code, C, K)

    elif decode.startswith('lp'):
        lp_type = None
        if decode.endswith('int'):
            lp_type = int
        decoder = lp.LPDecoder(code, dense=decode.startswith("lpdense"), lp_type=lp_type)

    elif decode=='exact':
        from qupy.ldpc.mps import ExactDecoder
        decoder = ExactDecoder(code)

    elif decode=='oe':
        from qupy.ldpc.mps import OEDecoder
        decoder = OEDecoder(code)

    elif decode=='logop':
        from qupy.ldpc.mps import LogopDecoder
        decoder = LogopDecoder(code)

    elif decode=='mps':
        from qupy.ldpc.mps import MPSDecoder
        decoder = MPSDecoder(code)

    elif decode=='mpsensemble':
        from qupy.ldpc.mps import MPSEnsembleDecoder
        decoder = MPSEnsembleDecoder(code)

    elif decode=='lookup':
        decoder = LookupDecoder(code)

    return decoder


def whack(code, decoder, p, N, C0, error_rate, mC, verbose, argv):
    """keep decoding until we get below error_rate"""

    errors = []
    succeeds = [] # put error ops here when we decode them
    fails = [] # if the decoder is wrong

    for i in range(N):
        err_op = ra.binomial(1, p, (code.n,))
        err_op = err_op.astype(numpy.int32)
        errors.append(err_op)

    target_succeed = (1. - error_rate) * N

    i = 0
    C = C0

    while len(succeeds) < target_succeed and len(succeeds)+len(errors) >= target_succeed:

        err_op = errors[i]

        # We use Hz to look at X type errors (bitflip errors)

        write(str(err_op.sum()))

        s = dot2(code.Hz, err_op)
        write(":s%d:"%s.sum())

        op = decoder.decode(p, err_op, C=C, verbose=verbose, argv=argv)

        c = 'F'
        success = False
        if op is None:
            print(i, "TRY AGAIN")
            i = i+1

        else:
            op = (op+err_op)%2
            # Should be a codeword of Hz (kernel of Hz)
            assert dot2(code.Hz, op).sum() == 0
            write("%d:"%op.sum())

            success = code.is_stab(op)
            c = '.' if success else 'x'

            errors.pop(i)

            if success:
                print(i, "GOOD")
                succeeds.append(err_op)
            else:
                print(i, "BAD")
                fails.append(err_op)

        if i==len(errors):
            i = 0
            C = int(ceil(C*mC))
            print(("C = ", C))

        write(c+' ')

        print("errors = ", len(errors), "succeeds =", len(succeeds), "fails =", len(fails))

    if len(succeeds) >= target_succeed:
        print("SUCCESS")

    else:
        assert len(succeeds)+len(errors) < target_succeed
        print("FAIL")

    print()
    print(datestr)
    print(argv)
    print("error rate = %.8f" % (1.*(len(errors) + len(fails))/N))
    print("fail rate = %.8f" % (1.*len(fails) / N))

if __name__ == "__main__":

    _seed = argv.get('seed')
    if _seed is not None:
        seed(_seed)
        ra.seed(_seed)

    if argv.profile:
        import cProfile as profile
        profile.run("main()")

    else:
        main()


