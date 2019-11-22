#!/usr/bin/env python3

import sys, os
from math import *
from random import *
import time

import numpy
import numpy.random as ra


from qupy.ldpc import solve
from qupy.ldpc.solve import shortstr
from qupy.ldpc.tool import save_alist
from qupy.argv import argv


dot = numpy.dot


class BPModel(object):
    "David MacKay: p560-561"

    def __init__(self, H, p, z):
        self.H = H
        self.p = p # bit flip probability
        self.z = z # syndrome
        m, n = H.shape
        self.m = m # rows (checks)
        self.n = n # cols (bits)

        # We are looking for x st. Hx = z

        # (q0, q1): bit n message to check m
        self.q0 = numpy.zeros((m, n), dtype=numpy.float64)
        self.q1 = numpy.zeros((m, n), dtype=numpy.float64)

        p0, p1 = 1-p, p
        self.q0[:] = H * p0 # i am zero
        self.q1[:] = H * p1 # i am one

    def horizontal(self):
        m, n = self.m, self.n

        # go through each check
#        for i in range(m):
#
#
#    def vertical(self):


    def step(self):
        self.horizontal()
        self.vertical()



class BaseDecoder(object):

    def __init__(self, d, H):
        self.d = d
        self.H = H
        m, n = H.shape
        self.m = m # rows
        self.n = n # cols

    def mkerr_weight(self, w):
        cols = range(self.n)
        err = numpy.zeros((self.n,), dtype=numpy.uint32)
        for i in range(w):
            col = choice(cols)
            cols.remove(col)
            err[col] = 1
        err = err.astype(numpy.uint32)
        write(" %de"%err.sum())
        return err

    def mkerr(self, p):
        assert 0.<=p<=1.
        while 1:
            err = ra.binomial(1, p, (self.n,))
#            err = numpy.zeros((self.n,), dtype=numpy.uint32)
#            err[randint(0, self.n-1)] = 1
            if err.sum() or 1:
                break
        err = err.astype(numpy.uint32)
        write(" %de"%err.sum())
        return err

    def check(self, err):
        #print self.H.shape, err.shape
        syndrome = dot(self.H, err) % self.d
        return syndrome


#from qupy import bayes
#
#class BrokenBPDecoder(BaseDecoder):
#    def build_model_1(self, p, syndrome):
#        H = self.H
#        factors = []
#        for row in range(self.m):
#            check = syndrome[row]
#            op = H[row, :]
#            vars = ['v%d'%col for col in range(self.n) if op[col]]
#            shape = (self.d,)*len(vars) 
#            P = numpy.zeros(shape, dtype=bayes.scalar)
#            size = self.d**len(vars)
#            for idx in xrange(size):
#                uidx = numpy.unravel_index(idx, shape)
#                value = sum(uidx) % self.d
#                #print idx, uidx, value
#                if value == check:
#                    P[uidx] = p**sum(uidx)
#                else:
#                    P[uidx] = 0.
#            #print check, list(P.ravel()), vars
#            P /= P.sum()
#            factor = bayes.Table(P, shape, vars)
#            factors.append(factor)
#        model = bayes.Model(factors)
#        return model
#
#    def build_model_2(self, p, syndrome):
#        H = self.H
#        factors = []
#        for row in range(self.m):
#            check = syndrome[row]
#            op = H[row, :]
#            vars = ['v%d'%col for col in range(self.n) if op[col]]
#            shape = (self.d,)*len(vars) 
#            P = numpy.zeros(shape, dtype=bayes.scalar)
#            size = self.d**len(vars)
#            for idx in xrange(size):
#                uidx = numpy.unravel_index(idx, shape)
#                value = sum(uidx) % self.d
#                #print idx, uidx, value
#                if value == check:
#                    P[uidx] = 1.
#                else:
#                    P[uidx] = 0.
#            #print check, list(P.ravel()), vars
#            P /= P.sum()
#            factor = bayes.Table(P, shape, vars)
#            factors.append(factor)
#        for col in range(self.n):
#            vars = ['v%d'%col]
#            shape = (self.d,)
#            P = numpy.zeros(shape, dtype=bayes.scalar)
#            P[0] = 1-p
#            P[1] = p
#            factor = bayes.Table(P, shape, vars)
#            factors.append(factor)
#        model = bayes.Model(factors)
#        return model
#
#    build_model = build_model_1
#
#    def init_algo(self, p, algo):
#        #write('[i')
#        messages = algo.messages
#        for factor in algo.factors:
#            for var in factor.vars:
#                m = messages[var, factor]
#                m[0] = 1-p
#                m[1] = p
#                assert m.shape == (2,)
#        #write(']')
#
#    def getop(self, algo):
#        op = []
#        for i in range(self.n):
#            v = 'v%d'%i
#            data = algo.marginal(v).data
#            idx = data.argmax()
#            op.append(idx)
#        return op
#
#    def decode(self, p, err, max_iter=None, verbose=False, **kw):
#        #print "decode:", err
#        syndrome = self.check(err)
#        #print "syndrome:", syndrome
#        write("b")
#        model = self.build_model(p, syndrome)
#        write("d")
#        algo = bayes.SumProduct(model)
#        #algo.init_random()
#        self.init_algo(p, algo)
#        if max_iter is None:
#            max_iter = self.n
#        count = 0
#        while count < max_iter:
#            #for i in range(4):
#            algo.step(1)
#            write(choice('/\\'))
#            op = self.getop(algo)
#            delta = (self.check(op) + syndrome) % self.d
#            if delta.sum() == 0:
#                #print "succeed!"
#                return op
#            count += 1
#        #print "fail"
#        #return op


class PyCodeBPDecoder(BaseDecoder):

    def __init__(self, d, H):
        self.d = d
        self.H = H
        m, n = H.shape
        self.m = m # rows
        self.n = n # cols

        from pycodes.pyLDPC import LDPCCode

        L = []
        for i in range(m):
            row = []
            for j in range(n):
                if H[i, j]:
                    row.append(j)
            L.append(row)

        #print L[:5], m, n

        #N = 25
        #L = L[:N]
        # count edges in code
        e = sum(len(z) for z in L)
        #print n, m, e, len(L)
        assert m == len(L)
        #m = len(L)
        #self.code = LDPCCode(n, n-m, e, L, m)
        self.code = LDPCCode(n, m, e, L, m)

    def decode(self, p, err, max_iter=None, verbose=False, **kw):
#        if verbose:
#            print
#            print "BPDecoder.decode:"
#            print shortstr(err)
#            print shortstr(self.check(err))
        code = self.code
        d = self.d
        assert d==2

        ev = [
            log(p/(1-p)) if x else log((1-p)/p)
            for x in err]
        #print min(ev)
        #from pycodes.utils.channels import BSC
        #print min(BSC([0]*len(ev), p))
        code.setevidence(ev, alg='SumProductBP')

        syndrome = self.check(err)
        if max_iter is None:
            max_iter = self.n
        count = 0
        while count < max_iter:
            #for i in range(4):
            #if verbose:
            #    print ' '.join(['%.2f'%x for x in code.getbeliefs()])
            code.decode()
#            if verbose:
#                print "badchecks:", code.badchecks()
            #write(choice('/\\'))
            op = [int(x>0.5) for x in code.getbeliefs()]

#            if verbose:
#                print shortstr(op)
#                #print ' '.join(['%.2f'%x for x in code.getbeliefs()])
#                print shortstr(self.check(op))

            if sum(op)==0:
                return (err+op)%2
#            delta = (self.check(op) + syndrome) % d
#            delta = delta.sum()
#            if delta == 0:
#                #print "succeed!"
#                return op
#            write('[%d]'%delta)
            count += 1
        #print "fail"
        #return op
    

class RadfordBPDecoder(BaseDecoder):

    def __init__(self, d, H):
        self.d = d
        self.H = H
        m, n = H.shape
        self.m = m # rows
        self.n = n # cols

        stem = 'tempcode_%.6d'%randint(0, 999999)
        save_alist(stem+'.alist', H)
        r = os.system('./alist-to-pchk -t %s.alist %s.pchk' % (stem, stem))
        assert r==0
        self.stem = stem

    def __del__(self):
        stem = self.stem
        for ext in 'alist pchk out'.split():
            try:
                os.unlink("%s.%s"%(stem, ext))
            except OSError:
                pass

    def decode(self, p, err, max_iter=None, verbose=False, **kw):
#        if verbose:
#            print
#            print "RadfordBPDecoder.decode:"
#            print shortstr(err)
#            syndrome = self.check(err)
#            print "syndrome:"
#            print shortstr(syndrome)

        d = self.d
        assert d==2

        stem = self.stem
        if max_iter is None:
            max_iter = argv.get("maxiter", self.n)

        try:
            os.unlink('%s.out'%stem)
        except:
            pass
        cmd = './decode %s.pchk - %s.out bsc %.4f prprp %d' % (
            stem, stem, p, max_iter)
        #print(cmd)
        #c_in = os.popen(cmd, 'w', 0)
        c_in = os.popen(cmd, 'w')
        #c_in, c_out = os.popen2(cmd, 'b', 0)

        for x in err:
            print(x, file=c_in)

        c_in.close()

        op = open('%s.out'%stem).read()

        #op = c_out.read()
        #print("[%s]" % repr(op), end="")
        op = [int(c) for c in op.strip()]
        syndrome = self.check(op)
#        if verbose:
#            print "result:"
#            print shortstr(op)
#            print "syndrome:"
#            print shortstr(syndrome)

        if syndrome.sum() == 0:
            return (err + op) % d


