#!/usr/bin/env python3

from random import *

import numpy
import numpy.random as ra

write = lambda s : print(s, end='', flush=True)


from qupy.ldpc import solve
from qupy.ldpc.solve import (
    pop2, zeros2, dot2, array2, eq2, rand2, binomial2,
    randexpo, shortstr, shortstrx)
            

class Decoder(object):
    def __init__(self, code):
        self.code = code

    def __getattr__(self, name):
        return getattr(self.code, name)


class SimpleDecoder(Decoder):
    def __init__(self, code):
        Decoder.__init__(self, code)
        code.build_all()

    def decode(self, p, err_op, verbose=False, **kw):
        #print "decode:"
        #print shortstr(err_op)
        T = self.get_T(err_op)
        #print shortstr(T)
        dist = self.get_dist(p, T)
        #print dist
        p1 = max(dist)
        idx = dist.index(p1)
        l_op = self.all_Lx[idx]
        #op = (l_op+T+err_op)%2
        op = (l_op+T)%2
        return op


class RandomDecoder(Decoder):
    def decode(self, p, err_op, verbose=False, **kw):
        #print "decode:"
        #print shortstr(err_op)
        T = self.get_T(err_op)
        #print shortstr(T)
        #l_op = choice(self.all_Lx)
        #op = (l_op+T+err_op)%2
        #op = (l_op+T)%2
        op = T
        return op


class MetroDecoder(Decoder):

#    def metropolis_k(self, p, T, M0, M1, K):
#        "M0: warmup, M1: sample, K: subsample"
#        count = 0
#        T0 = T
#        n0 = T0.weight
#
#        assert M1//K
#        dist = []
#        while count<M0+M1:
#            stab = choice(self.stab_x)
#            T1 = stab*T0
#            n1 = T1.weight
#            r = (p/(1-p))**(n1-n0)
#            if r >= 1 or random() <= r:
#                T0 = T1
#                n0 = n1
#            if count>M0:
#                if count%K==0:
#                    dist.append(p**n0)
#            count += 1
#        #print "metropolis_k:", dist
#        return sum(dist)/len(dist)

    def metropolis(self, p, T, N, Hx=None):

        if Hx is None:
            Hx = self.Hx
        m, n = Hx.shape
        count = 0
        T0 = T
        n0 = T.sum()
        while count<N:

            idx = randint(0, m-1)
            stab = Hx[idx]
            T1 = (stab + T0)%2
            n1 = T1.sum()

            r = (p/(1-p))**(n1-n0)
            if r >= 1 or random() <= r:
                T0[:] = T1 # <-- modify inplace
                n0 = n1

            count += 1

        return n0

    def decode(self, p, err_op, verbose=False, **kw):

        from qupy.ldpc.metro import metropolis

        #print "decode:"
        strop = self.strop
        #print strop(err_op)
        T = self.get_T(err_op)

        all_Lx = list(solve.span(self.Lx))

        M0 = kw.get('M0', 100000)
        best_l = None
        best_q = -self.n
        best_i = None
        best_T = None
        #print
        #print "T:"
        #print strop(T)
        for i, l_op in enumerate(all_Lx):
            #print "l_op:"
            #print strop(l_op)
            T1 = (T+l_op)%2
#            for h in self.Hx:
#                if random()<0.5:
#                    T1 += h
#                    T1 %= 2
            #q = -self.metropolis(p, T1, N)
            q = -metropolis(p, T1, M0, self.Hx)
            #write("(%d)"%q)
            #print "T %d:"%i
            #print strop(T1)
            if q > best_q:
                best_l = l_op
                best_q = q
                best_i = i
                best_T = T1
            #print "_"*79
        #write(":%d "%best_i)
        #print "best_T"
        #print strop(best_T)
        T += best_l
        T %= 2
        #print "T"
        #print strop(T)
        return T


class TemperedDecoder(MetroDecoder):

    def decode(self, p, err_op, verbose=False, **kw):


        Hx = self.Hx
        Lx = self.Lx
        T = self.get_T(err_op)

        Nc = kw.get('Nc', 20)
        M0 = kw.get('M0', 1000)
        M1 = kw.get('M1', 100)

        metropolis(p, T, 1000*M0, self.Hx)

        Ts = [T.copy() for ci in range(Nc)]

        delta = (0.75 - p)/(Nc-1)

        m, n = Hx.shape
        HxLx = zeros2(m+Lx.shape[0], n)
        HxLx[:m] = Hx
        HxLx[m:] = Lx

        marks = [0] * Nc

        for i in range(M1):

            marks[Nc-1] |= 1 # top guy

            for ci in range(Nc):
                p_m = p + ci*delta # modified p

                ops = Hx if ci+1<Nc else HxLx
                #self.metropolis(p_m, Ts[ci], M0, ops)
                metropolis(p_m, Ts[ci], M0, ops)

            for ci in range(Nc-1):
                p_m = p + ci*delta 
                p_m_1 = p + (ci+1)*delta 
                r = ((p_m*(1-p_m_1))/(p_m_1*(1-p_m)))**(
                    Ts[ci+1].sum() - Ts[ci].sum())
                if r>=1 or random()<=r:
                    #write("[%d]"%ci)
                    Ts[ci+1], Ts[ci] = Ts[ci], Ts[ci+1]
                    marks[ci+1], marks[ci] = marks[ci], marks[ci+1]

            if kw.get("show"):
                print("%5d"%i, end=' ')
                print(''.join((str(m)[-1] if m else '.') for m in marks), end=' ')
                print("%.4f"%(1.*sum(marks)/n))

            if marks[0]:
                marks[0] |= 2 # this guy made it to the bottom

        tops0 = 0
        for mark in marks:
            if mark&2:
                tops0 += 1

        write("(%d)"%tops0)

        return Ts[0]



class StarMetroDecoder(MetroDecoder):

    def decode(self, p, err_op, verbose=False, **kw):
        T = self.get_T(err_op)
        Lx = self.Lx
        k = Lx.shape[0]
        Ts = [T] + [(l_op+T)%2 for l_op in Lx]
        n = len(Ts)

#        for i in range(n):
#          T = Ts[i]
#          for s_x in self.Hx:
#            if random()<0.5:
#              T += s_x
          #for l_x in Lx:
          #  if random()<0.5:
          #    T += l_x
#          T %= 2

        N = 100000
        M = 100

        logops = [zeros2(self.n)] + list(Lx)

        #print
        count = 0
        while count < M:
            qs = []
            for i in range(n):
                q = -metropolis(p, Ts[i], N)
                qs.append(q)

            q = max(qs)
            idx = qs.index(q)
            #print qs, idx
            if idx==0:
                break
            l_op = logops[idx] # <-- best so far
            # swap out the best idx
            logops[0], logops[idx] = logops[idx], logops[0]
            Ts[0], Ts[idx] = Ts[idx], Ts[0]
            for i in range(1, n):
                if i!=idx:
                    Ts[i] += l_op
                    Ts[i] %= 2
            count += 1

        #print "idx =", idx
        #print shortstr(Ts[0])

        return Ts[0]


class DynamicDecoder(Decoder):
    def __init__(self, code):
        Decoder.__init__(self, code)
        self.all_Lx = list(solve.span(self.Lx))
        self.graph = Tanner(self.Hx)

    def decode(self, p, err_op, verbose=False, **kw):
        T = self.get_T(err_op)
        all_Lx = self.all_Lx
        Hx = self.Hx
        graph = self.graph

        best_T = T
        best_w = T.sum()
        for i, l_op in enumerate(all_Lx):
            T1 = (T+l_op)%2
            for h in Hx:
                if random()<0.5:
                    T1 += h
                    T1 %= 2
            T1 = graph.localmin(T1, verbose=False)
            w = T1.sum()
            if w < best_w:
                best_T = T1
                best_w = w

        return best_T


class SchoningDecoder(Decoder):

    """
    Inspired by Schoning's algorithm: 
    Randomly flip a bit to turn off a random stabilizer. 
    Stop if all stabilizers are off.
    Start again after some fixed interval.
    """

    def decode(self, p, err_op, M0=10, verbose=False, **kw):
        Hz = self.Hz
        m, n = Hz.shape

        #while 1:
        for i in range(M0):

            T = zeros2(n)

            #T = ra.binomial(1, p, (n,))
            #T = T.astype(numpy.int32)
    
            #fails = [True] * m
    
            count = 0
            #while count < 3*n:
            while count < 100:
                count += 1
    
                v = dot2(Hz, (T+err_op))
    
                #write('%s '%v.sum())
    
    #            print
    #            print shortstr(v)
    
                rows = numpy.where(v)[0]
                if not len(rows):
                    return T
    
                # Pick random stabilizer
                row = choice(rows)
    #            print rows, row
    
                # Flip random bit in this stabilizer (to turn it off):
                cols = numpy.where(Hz[row])[0]
                col = choice(cols)
    
                T[col] = 1 - T[col]




class StarDynamicDecoder(Decoder):
    def __init__(self, code):
        Decoder.__init__(self, code)
        Hx, Lx = self.Hx, self.Lx
        H = zeros2(Lx.shape[0]+Hx.shape[0], self.n)
        H[:Lx.shape[0]] = Lx
        H[Lx.shape[0]:] = Hx
        self.graph = Tanner(H)

    def decode(self, p, err_op, verbose=False, **kw):
        T = self.get_T(err_op)
        Hx = self.Hx
        Lx = self.Lx
        graph = self.graph

        M0 = kw.get("M0", 10)

        best_T = T
        best_w = T.sum()
        for i in range(M0):
            #print "l_op:"
            #print strop(l_op)
            T1 = T.copy()
            for op in Hx:
                if random()<0.5:
                    T1 += op
                    T1 %= 2
            for op in Lx:
                if random()<0.5:
                    T1 += op
                    T1 %= 2
            T1 = graph.localmin(T1, verbose=False)
            w = T1.sum()
            if w < best_w:
                best_T = T1
                best_w = w

        return best_T


def minweight(op, Hx):

    m, n = Hx.shape
    w = op.sum()
    changed = True
    while changed and w>0:
        changed = False
        for i in range(m):
            op1 = (op + Hx[i])%2
            w1 = op1.sum()
            if w1<w:
                op = op1
                w = w1
                changed = True
    return op
            


class PMADecoder(Decoder):

    # XXX this seems totally hopeless.. XXX

    def __init__(self, code):
        Decoder.__init__(self, code)
        self.all_Lx = list(solve.span(self.Lx))

    def build_graph(self):

        Hx = self.Hx
        Tx = self.Tx

        # Build edge weights
        n = self.n
        edges = []
        graph = {}
        op = zeros2(n)
        for i in range(n):
          for j in range(i+1, n):
            op[i] = 1
            op[j] = 1
            T = self.get_T(op)
            w0 = T.sum()
            T = minweight(T, Hx)
            w1 = T.sum() # could also use a metropolis weight...
            edges.append(w1)
            graph[i, j] = w1
            #print "%d:%d"%(n, T.sum()),
            op[i] = 0
            op[j] = 0
        #print

        # Build boundary weights
        bdy = []
        m = Tx.shape[0]
        for i in range(m):
            w0 = Tx[i].sum()
            T = minweight(Tx[i], Hx)
            w1 = T.sum()
            if w1<w0:
                Tx[i] = T
                w0 = w1
            bdy.append(w0)
            graph[i] = w0
            #print "%d:%d"%(w0, w1),

        self.edges = edges
        self.bdy = bdy
        self.graph = graph

    def weight(self, p, T, Hx, **kw):

        T = T.copy()
        T = minweight(T, Hx)
        w0 = T.sum()
        w = w0+1

        M0 = kw.get("M0", 100000)

        m, n = Hx.shape
        while w>w0:

            for i in range(m):
                if random()<0.5:
                    T = (T+Hx[i])%2

            w = metropolis(p, T, M0, Hx)
            M0 += 10000

            break

        return w

    def homweight(self, p, T):
        all_Lx = self.all_Lx
        Hx = self.Hx

        w0 = self.weight(p, T, Hx)
        best_T = T
        for lop in all_Lx:

            T1 = (T+lop)%2
            w1 = self.weight(p, T1, Hx)

            if w1 < w0:
                best_T = T1
                w0 = w1
        return w1

    def metric1(self, p, i):
        T = self.Tx[i]
        return self.homweight(p, T)

    def metric2(self, p, i, j):
        T = (self.Tx[i] + self.Tx[j])%2
        return self.homweight(p, T)

    def decode(self, p, err_op, verbose=False, **kw):

        n = self.n
        idxs = [] # defects
        for i in range(n):
            if err_op[i]:
                idxs.append(i)

        graph = []
        k = len(idxs)
        for i in range(k):
          for j in range(i+1, k):
            w = self.metric2(p, i, j)
            graph.append((i, j, w))
        for i in range(k):
            w = self.metric1(p, i)
            graph.append((i, k, w))

        print(graph)




class MLDecoder(Decoder):
    "Inspired by k-means clustering. Does not scale well with code.n"

    def __init__(self, code):
        Decoder.__init__(self, code)

        syndromes = []
        errors = []
        for i in range(code.n+1):
            error = zeros2(code.n)
            if i<code.n:
                error[i] = 1
            syndrome = dot2(code.Hz, error)
            syndromes.append(syndrome)
            errors.append(error)

        #for i in range(code.n):
        #  for j in range(i+1, code.n):
        #    error = zeros2(code.n)
        #    error[i] = 1
        #    error[j] = 1
        #    syndrome = dot2(code.Hz, error)
        #    syndromes.append(syndrome)
        #    errors.append(error)
    
        self.xs = array2(syndromes)
        self.ys = array2(errors)

    search = [] 
    for n in range(1, 1000):
        items = [] 
        n1 = n
        i = 0
        while n1:
            if n1%2:
                items.append(i)
            i += 1
            n1 >>= 1
        search.append(items)

    def decode(self, p, err_op, verbose=False, **kw):

        code = self.code
        xs, ys = self.xs, self.ys

        x = dot2(code.Hz, err_op)
        x.shape = (1, code.mz)
        a = (self.xs + x)%2
        #print shortstr(a)

        a = a.sum(1)

        a = list(enumerate(a))
        a.sort(key = lambda item : item[1])
#        print a

#        print shortstr(y)
#        print shortstr(x)

        for idxs in self.search:
            #print idxs
            y1 = ys[a[idxs[0]][0]]
            for idx in idxs[1:]:
                y1 = y1 + ys[a[idx][0]]
#            print shortstr(y1)
            x1 = dot2(code.Hz, y1)
            #print shortstr(x1)
            if eq2(x1, x):
#                print "OK", idxs
                break
        else:
            return None

        return y1


class MultiDecoder(Decoder):
    """
    """

    def __init__(self, code, decode1, decode2):
        Decoder.__init__(self, code)
        self.decode1 = decode1
        self.decode2 = decode2

    def decode(self, p, err_op, verbose=False, **kw):

        op = self.decode1.decode(p, err_op, verbose=verbose, **kw)

        if op is None:

            op = self.decode2.decode(p, err_op, verbose=verbose, **kw)

        return op




