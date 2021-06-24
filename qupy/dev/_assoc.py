#!/usr/bin/env python3

import numpy
import scipy
from scipy import optimize

from qupy.dev import linalg


EPSILON = 1e-8


class AssocAlg(object):
    "_associative algebra over (python) complex _numbers"

    def __init__(self, struct, sub=None):
        N = struct.shape[0]
        assert struct.shape == (N, N, N)
        self.struct = struct
        self.N = N
        basis = []
        for i in range(N):
            v = [0.]*N
            v[i] = 1.
            basis.append(v)
        self.basis = numpy.array(basis)
        self.unit = self.find_unit()
        self.sub = sub # basis in some bigger algebra

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        N = self.N
        v = numpy.zeros((N,), dtype=float)
        v[idx] = 1.
        return v

    def __str__(self):
        return "AssocAlg(dim=%d)"%(self.N,)

    def mul(self, u, v):
        assert len(u) == self.N
        assert len(v) == self.N
        c = self.struct
        vc = numpy.dot(v, c)
        w = numpy.dot(u, vc)
        assert w.shape == (self.N,)
        return w

    def find_unit(self):
        N, struct = self.N, self.struct
        rows = []
        rhs = []
        for i in range(N):
          for k in range(N):
            rhs.append(int(i==k))
            rows.append(struct[:, i, k])
        A = numpy.array(rows)
        b = numpy.array(rhs)
        Ainv = numpy.linalg.pinv(A)
        I = numpy.dot(Ainv, b)
        #print(I)
        assert numpy.linalg.norm(I) > EPSILON, str(I)
        return I

    def check(self):
        # check unit
        I = self.unit
        for u in self.basis:
            v = self.mul(u, I)
            assert numpy.allclose(v, u)
            v = self.mul(I, u)
            assert numpy.allclose(v, u)
        N, struct = self.N, self.struct
        # check assoc ... todo

    def is_commutative(self):
        c = self.struct
        return numpy.allclose(c, c.transpose(1, 0, 2))
        #mul = self.mul
        #for u in self.basis:
        #  for v in self.basis:
        #    if not numpy.allclose(mul(u, v), mul(v, u)):
        #        assert not numpy.allclose(c, c.transpose(1, 0, 2))
        #        return False
        #assert numpy.allclose(c, c.transpose(1, 0, 2))
        #return True

    def construct(self, x, basis=None):
        "construct x as a vector in a bigger algebra"
        if basis is None:
            basis = self.sub
        assert basis is not None
        assert len(basis)
        assert len(x) == len(basis) == self.N
        v = 0. * basis[0]
        for i, a_i in enumerate(x):
            if abs(a_i)<EPSILON:
                continue
            opi = a_i * basis[i]
            if v is None:
                v = opi
            else:
                v = v + opi
        return v

    def subalgebra(self, basis):
        basis = numpy.array(basis)
        assert basis.shape[1] == self.N
        E = basis.transpose()
        Einv = numpy.linalg.pinv(E, EPSILON)
        #print(E)
        #print(Einv)
        #print(numpy.dot(Einv, E))
    
        N = len(basis)
        struct = numpy.zeros((N, N, N))
        for i, u in enumerate(basis):
          for j, v in enumerate(basis):
            w = self.mul(u, v)
            cij = numpy.dot(Einv, w)
            struct[i, j] = cij
        A = AssocAlg(struct, sub=basis)
        return A

    def find_center(self):
        "find basis for the center Z(A)."
        N, c = self.N, self.struct
        cT = c.transpose(1, 0, 2)
        A = c - cT
        A.shape = (N, N*N)
        A = A.transpose()
        B = linalg.kernel(A)
        return B.transpose()

    def find_centralizer(self, x):
        N, struct = self.N, self.struct
        rows = []
        for k in range(N):
            cij = struct[:, :, k]
            xicij = numpy.dot(x, cij)
            xjcij = numpy.dot(x, cij.transpose())
            rows.append(xjcij - xicij)
        A = numpy.array(rows)
        assert sum(abs(ui) for ui in numpy.dot(A, x)) < EPSILON
        assert sum(abs(ui) for ui in numpy.dot(A, self.unit)) < EPSILON
        a = linalg.kernel(A)
        assert a.shape[0] == N
        a = a.transpose()
        return a

    def principal_ideal(self, x):
        N, c = self.N, self.struct
        assert len(x) == N
        basis = []
        for y in self.basis:
            xy = self.mul(x, y)
            basis.append(xy)
        basis = linalg.row_reduce(basis, truncate=True)
        A = self.subalgebra(basis)
        return A

    def mod(self, basis):
        "mod out by ideal spanned by basis"
        #print("mod", self.N, len(basis))
        E = numpy.array(basis)
        assert E.shape[1] == self.N
        Et = E.transpose()
        Etinv = numpy.linalg.pinv(Et, EPSILON)
        P = numpy.dot(Et, Etinv)
        assert numpy.allclose(P, numpy.dot(P, P))
        I = numpy.identity(self.N)
        P = I-P
        assert numpy.allclose(P, numpy.dot(P, P))
        #print(P)
        basis = []
        for e in self.basis:
            e = numpy.dot(P, e)
            basis.append(e)
        basis = linalg.row_reduce(basis)
        N = len(basis)
        #print("N =", len(basis))
        #print(basis)
        E = basis.transpose()
        Einv = numpy.linalg.pinv(E, EPSILON)
        struct = numpy.zeros((N, N, N))
        for i, u in enumerate(basis):
          for j, v in enumerate(basis):
            w = self.mul(u, v)
            w = numpy.dot(P, w)
            cij = numpy.dot(Einv, w)
            struct[i, j] = cij
        #print(struct)
        A = AssocAlg(struct)
        return A

    def find_idempotents(self, trials=None, verbose=False):
        # solve for idempotents in the algebra of invariants
        # see: https://gist.github.com/nicoguaro/3c964ae8d8f11822b6c22768dc9b3716

        if trials==0:
            return []

        N, struct = self.N, self.struct

        inbls = ["a_%d"%i for i in range(N)]
        outbls = ["b_%d"%i for i in range(N)]
        lines = ["def func(X):"]
        lines.append("    %s = X"%(', '.join(v for v in inbls)))
        for k in range(N):
            terms = []
            for i in range(N):
              for j in range(i, N):
                if i==j:
                    val = struct[i, j, k]
                else:
                    val = struct[i, j, k] + struct[j, i, k]
                if abs(val)>EPSILON:
                    terms.append("%s*%s*%s" % (val, inbls[i], inbls[j]))
            terms.append("-%s"%inbls[k])
            line = "    %s = %s" % (outbls[k], ' + '.join(terms))
            line = line.replace("+ -", "- ")
            lines.append(line)
        lines.append("    return [%s]" % (', '.join(v for v in outbls)))
        stmt = '\n'.join(lines)
        if verbose:
            print(stmt)
        ns = {}
        exec(stmt, ns, ns)
        func = ns["func"]
    
        lines = ["def jacobian(X):"]
        lines.append("    %s = X"%(', '.join(v for v in inbls)))
        for k in range(N):
            comps = []
            for l in range(N): # diff with respect to a_l
                terms = []
                val = struct[k, k, k]
                for i in range(N):
                  for j in range(i, N):
                    if i==j:
                        val = struct[i, j, k]
                    else:
                        val = struct[i, j, k] + struct[j, i, k]
                    if abs(val)<EPSILON:
                        continue
                    if i==j==l: # a_l * a_l
                        terms.append("%s*%s" % (val, inbls[l]))
                    elif i==l:  # a_l * a_j
                        terms.append("%s*%s" % (val, inbls[j]))
                    elif j==l:  # a_i * a_l
                        terms.append("%s*%s" % (val, inbls[i]))
                total = '+'.join(terms)
                if l==k:
                    total += "-1.0"
                total = total.replace("+-", "-")
                comps.append(total)
            lines.append("    %s = [%s]"%(outbls[k], ', '.join(comps)))
        lines.append("    return [%s]" % (', '.join(outbls)))
        stmt = '\n'.join(lines)
    
        #print(stmt)
        ns = {}
        exec(stmt, ns, ns)
        jacobian = ns["jacobian"]
    
        #def check_jac(func, jacobian, x0):
        #    assert N==len(x0)
        #    for i in range(N):
    
        tol = 1e-10
        xs = []

        i = 0
        while 1:
            x0 = numpy.random.uniform(-0.1, 0.1, (N,))
            #print(x0)
            #print(func(x0))
    
            # FAIL: this only finds *real* solutions...

            method = "lm"
            #sol = optimize.root(func, x0, (), method, jacobian, tol) # broken...
            sol = optimize.root(func, x0, (), method, None, tol)
            assert sol.success, sol.message
            #print(sol.message)
            # nfev, njev, success
    
            #print("func:", func(sol.x))
            #print("jacobian:", jacobian(sol.x))
        
            if (sum(abs(xx) for xx in sol.x) > EPSILON and 
                not numpy.allclose(sol.x, self.unit)):
                yield sol.x

            i += 1

            if trials and i>=trials:
                break

    def find_reflectors(self, trials=None, verbose=False):
        if trials==0:
            return []

        N, struct = self.N, self.struct
        unit = self.unit

        inbls = ["a_%d"%i for i in range(N)]
        outbls = ["b_%d"%i for i in range(N)]
        lines = ["def func(X):"]
        lines.append("    %s = X"%(', '.join(v for v in inbls)))
        for k in range(N):
            terms = []
            for i in range(N):
              for j in range(i, N):
                if i==j:
                    val = struct[i, j, k]
                else:
                    val = struct[i, j, k] + struct[j, i, k]
                if abs(val)>EPSILON:
                    terms.append("%s*%s*%s" % (val, inbls[i], inbls[j]))
            terms.append("-%s"%unit[k])
            line = "    %s = %s" % (outbls[k], ' + '.join(terms))
            line = line.replace("+ -", "- ")
            line = line.replace("- -", "+ ")
            lines.append(line)
        lines.append("    return [%s]" % (', '.join(v for v in outbls)))
        stmt = '\n'.join(lines)
        if verbose:
            print(stmt)
        ns = {}
        exec(stmt, ns, ns)
        func = ns["func"]
    
        tol = 1e-10
        xs = []

        i = 0
        while 1:
            x0 = numpy.random.uniform(-0.1, 0.1, (N,))
            #print(x0)
            #print(func(x0))
    
            # FAIL: this only finds *real* solutions...

            method = "lm"
            #sol = optimize.root(func, x0, (), method, jacobian, tol) # broken...
            sol = optimize.root(func, x0, (), method, None, tol)
            assert sol.success, sol.message
            #print(sol.message)
            # nfev, njev, success
    
            #print("func:", func(sol.x))
            #print("jacobian:", jacobian(sol.x))
        
            if (not numpy.allclose(sol.x, unit) and 
                not numpy.allclose(sol.x, -unit)):
                yield sol.x

            i += 1

            if trials and i>=trials:
                break




