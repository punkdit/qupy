#!/usr/bin/env python3

import sys, os
from math import *
from random import *
import time

import numpy
import numpy.random as ra
from numpy.linalg import lstsq

#from qupy.tool.smap import SMap
from qupy.ldpc import solve
from qupy.ldpc.solve import (
    shortstr, shortstrx, hbox,
    eq2, dot2, compose2, rand2,
    pop2, insert2, append2, 
    array2, zeros2, identity2, parse)
from qupy.ldpc.tool import write #, load, save
from qupy.ldpc.chain import Chain, Morphism, equalizer

#from qupy.ldpc.decoder import Decoder, RandomDecoder
##from qupy.ldpc.dynamic import Tanner
#from qupy.ldpc.bpdecode import RadfordBPDecoder
#from qupy.ldpc.cluster import ClusterCSSDecoder
##from qupy.ldpc import ensemble
##from qupy.ldpc import lp

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
    assert C.sum() == 0 #, "\n%s"%shortstr(C)


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
            check=True, verbose=True, logops_only=False):

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
            self.build(check=check, logops_only=logops_only)
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

    def build(self, logops_only=False, check=True, verbose=False):
    
        Hx, Hz = self.Hx, self.Hz
        Lx, Lz = self.Lx, self.Lz
        Tx, Tz = self.Tx, self.Tz

        if verbose:
            _write = write
        else:
            _write = lambda *args : None
    
        _write('li:')
        self.Hx = Hx = solve.linear_independent(Hx)
        self.Hz = Hz = solve.linear_independent(Hz)
    
        mz, n = Hz.shape
        mx, nx = Hx.shape
        assert n==nx
        assert mz+mx<=n, (mz, mx, n)
    
        _write('build:')
    
        if check:
            # check kernel of Hx contains image of Hz^t
            check_commute(Hx, Hz)
    
        if Lz is None:
            _write('find_logops(Lz):')
            Lz = solve.find_logops(Hx, Hz, verbose=verbose)
            #print shortstr(Lz)
            #_write(len(Lz))

        k = len(Lz)
        assert n-mz-mx==k, "_should be %d logops, found %d. Is Hx/z degenerate?"%(
            n-mx-mz, k)

        _write("n=%d, mx=%d, mz=%d, k=%d\n" % (n, mx, mz, k))
    
        # Find Lx --------------------------
        if Lx is None:
            _write('find_logops(Lx):')
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

        if not logops_only:

            # Find Tz --------------------------
            _write('Find(Tz):')
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
            _write('Find(Tx):')
            U = zeros2(n, n)
            U[:mz] = Hz
            U[mz:mz+k] = Lz
            U[mz+k:] = Tz
    
            B = zeros2(n, mz)
            B[:mz] = identity2(mz)
            Tx_t = solve.solve(U, B)
            Tx = Tx_t.transpose()
    
            _write('\n')
        
            if check:
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
        #write("checking...")
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
        #write("done\n")

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

    def glue(self, i1, i2):
        assert i1!=i2

        Hx = self.Hx
        Hz = self.Hz
        mx, n = Hx.shape
        mz, _ = Hz.shape
        k = 1 
    
        A = Chain([Hz, Hx.transpose()])
        C  = Chain([identity2(k), zeros2(k, 0)])
    
        fn = zeros2(n, 1)
        fn[i1, 0] = 1 
        fm = dot2(Hz, fn) 
        f = Morphism(C, A, [fm, fn, zeros2(mx, 0)])
    
        gn = zeros2(n, 1)
        gn[i2, 0] = 1 
        gm = dot2(Hz, gn) 
        g = Morphism(C, A, [gm, gn, zeros2(mx, 0)])
    
        _, _, D = equalizer(f, g)
    
        Hz, Hxt = D[0], D[1]
        Hx = Hxt.transpose()
        code = CSSCode(Hx=Hx, Hz=Hz)
        return code

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

