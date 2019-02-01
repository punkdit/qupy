#!/usr/bin/env python3

"""
parse IrreducibleRepresentations output from gap.
"""

import sys, os
from time import sleep
import select
from subprocess import Popen, PIPE

import numpy

from qupy.dense import Qu
from qupy.util import mulclose
from qupy.argv import argv


class Gap(object):
    "IO interface to gap"
    def __init__(self):
        self.proc = Popen("gap", bufsize=0, stdin=PIPE, stdout=PIPE)
        sleep(0.1)
        #print(dir(proc.stdout))
        #print(proc.stdout.read(20))
        self.buf = ""
        self.pos = 0

    def read_nb(self):
        proc = self.proc
        data = bytes()
        while select.select([proc.stdout],[],[],0)[0]!=[]:   
            data += proc.stdout.read(1)
        data = data.decode("utf-8")
        self.buf += data

    def send(self, data):
        proc = self.proc
        data = data + "\n"
        data = data.encode("utf-8")
        proc.stdin.write(data)

    def expect(self, s):
        while s not in self.buf[self.pos:]:
            self.read_nb()
            sleep(0.1)

        pos = self.pos + self.buf[self.pos:].index(s)
        data = self.buf[self.pos : pos]
        self.pos = pos + len(s)
        return data



class Irrep(object):

    def __init__(self, gen, els, ops):
        n = ops.shape[0] # number of generators
        assert n == len(gen)
        degree = ops.shape[1]
        assert ops.shape[2] == degree
        ops = [Qu((degree, degree), 'ud', v) for v in ops]
        assert len(ops) == n
        self.ops = ops
        lookup = dict(zip(gen, ops))
        #self.group = mulclose(ops)
        lookup["I"] = Qu.identity(degree)

        for el in els:
            op = eval(el, lookup)
            print("%s =" % el)
            print(op)



def E(n):
    return numpy.exp(2*numpy.pi*1.j/n)


def parse_els(s):
    """
    example string:
    [ <identity> of ..., f1, f2, f3, f4, f5, f1*f2, f1*f3, f1*f4, f1*f5, f2^2, f2*f3, f2*f4, 
  f2*f5, f3*f4, f3*f5, f4*f5, f1*f2^2, f1*f2*f3, f1*f2*f4, f1*f2*f5, f1*f3*f4, f1*f3*f5, 
  f1*f4*f5, f2^2*f3, f2^2*f4, f2^2*f5, f2*f3*f4, f2*f3*f5, f2*f4*f5, f3*f4*f5, f1*f2^2*f3, 
  f1*f2^2*f4, f1*f2^2*f5, f1*f2*f3*f4, f1*f2*f3*f5, f1*f2*f4*f5, f1*f3*f4*f5, f2^2*f3*f4, 
  f2^2*f3*f5, f2^2*f4*f5, f2*f3*f4*f5, f1*f2^2*f3*f4, f1*f2^2*f3*f5, f1*f2^2*f4*f5, 
  f1*f2*f3*f4*f5, f2^2*f3*f4*f5, f1*f2^2*f3*f4*f5 ]
    """

    assert "f" in s, s

    s = s.replace("\n", "")
    s = s.strip()
    
    assert s[:2] == "[ "
    assert s[-2:] == " ]"
    s = s[2:-2]

    s = s.replace(" ", "")
    s = s.replace("^", "**")

    els = s.split(',')
    els = ['I' if el.startswith("<identity>") else el for el in els]

    return els


def scan_brackets(s, idx):
    count = 0
    while idx < len(s):
        if s[idx] == "[":
            count += 1
        elif s[idx] == "]":
            count -= 1
        idx += 1
        if count == 0:
            return idx
    return idx


def parse_irreps(s, els):
    s = s.replace("\n", " ")
    s = s.strip()
    assert s[:2] == "[ "
    assert s[-2:] == " ]"
    s = s[2:-2]

    s = s.replace(" ", "")

    s = s.replace("^", "**")

    irreps = []

    gen = None

    idx = 0
    while idx < len(s):
        assert s[idx] == "["
        jdx = scan_brackets(s, idx)
        lhs = s[idx:jdx]
        assert lhs[0] == "["
        assert lhs[-1] == "]"
        lhs = lhs[1:-1]
        lhs = lhs.split(',')
        assert gen is None or lhs==gen
        gen = lhs

        idx = jdx
        if s[idx:idx+2] == "->":
            idx += 2
        else:
            break

        assert s[idx] == "["
        jdx = scan_brackets(s, idx)
        rhs = s[idx: jdx]
        #print(rhs)
        rhs = eval(rhs)
        rhs = numpy.array(rhs)
        #print(rhs)
        print(rhs.shape)

        irrep = Irrep(gen, els, rhs)
        irreps.append(irrep)

        if jdx == len(s):
            break
        
        idx = jdx + 1

    



def main(name):

    (i, j) = {
        "Tetra" : (24, 3),  # IdSmallGroup(SL(2,3));
        "Octa" : (48, 28),  
        "Icosa" : (120, 5), # IdSmallGroup(SL(2,5));
    }[name]

    assert name != "Icosa", "not implemented" # represented as a perm group... :P
    
    gap = Gap()
    PROMPT = "gap> "
    gap.expect(PROMPT)
    
    gap.send("G:=SmallGroup(%d, %d);" % (i,j))

    s = gap.expect(PROMPT)
    
    gap.send("Elements(G);")

    s = gap.expect(PROMPT)

    els = parse_els(s)
    print(els)

    gap.send("IrreducibleRepresentations(G);")
    
    s = gap.expect(PROMPT)
    
    irreps = parse_irreps(s, els)


main(argv.next())




