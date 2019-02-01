#!/usr/bin/env python3

import numpy
import scipy
from scipy import optimize

from qupy.dev import linalg


EPSILON = 1e-8


class AssocAlg(object):
    "_associative algebra over (python) complex _numbers"

    def __init__(self, struct):
        N = struct.shape[0]
        assert struct.shape == (N, N, N)
        self.struct = struct
        self.N = N
        self.unit = self.find_unit()

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
        x = numpy.dot(Ainv, b)
        #print(x)
        assert numpy.linalg.norm(x) > EPSILON, str(x)
        return x

    def find_center(self):
        "find basis for the center Z(A)."
        N, c = self.N, self.struct
        cT = c.transpose(1, 0, 2)
        A = c - cT
        A.shape = (N, N*N)
        A = A.transpose()
        B = linalg.kernel(A)
        return B.transpose()

    def construct(self, x, basis):
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

    def check_assoc(self):
        N, struct = self.N, self.struct


    def find_idempotents(self, trials=1):
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
        #print(stmt)
        exec(stmt, locals(), globals())
        assert func
    
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
        exec(stmt, locals(), globals())
        assert jacobian
    
        #def check_jac(func, jacobian, x0):
        #    assert N==len(x0)
        #    for i in range(N):
    
        tol = 1e-10
        xs = []
        for i in range(trials):
            x0 = numpy.random.uniform(-1, 1, (N,))
    #    for i in range(N):
    #        x0 = numpy.zeros((N,))
    #        x0[i] = 1.
    
            method = "lm"
            #sol = optimize.root(func, x0, (), method, jacobian, tol) # broken...
            sol = optimize.root(func, x0, (), method, None, tol)
            assert sol.success, sol.message
            #print(sol.message)
            # nfev, njev, success
    
            #print("func:", func(sol.x))
            #print("jacobian:", jacobian(sol.x))
        
            xs.append(sol.x)
        return xs



