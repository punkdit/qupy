#!/usr/bin/env python3

import sys
import math
from operator import mul, matmul

import numpy

from qupy.dense import Qu, Gate, Vector
from qupy.dense import genidx, bits, is_close, on, off, scalar
from qupy.dense import commutator, anticommutator
from functools import reduce

def cleanup(diagram):
    lines = diagram.split('\n')
    lines = [l for l in lines if l.strip()]
    start = None
    for line in lines:
        for idx, c in enumerate(line):
            if c!=' ':
                start = min(start, idx) if start is not None else idx
                break
    lines = [l[start:]+'\n' for l in lines]
    return ''.join(lines)


class ParseError(Exception):
    def __init__(self, matrix, row, col, desc='ParseError'):
        self.matrix = matrix
        self.row = row
        self.col = col
        self.desc = desc

    def __str__(self):
        matrix = self.matrix.clone()
        for row in ([self.row] if type(self.row) is int else self.row):
            matrix[row, self.col] = '?'
        desc = "%s:\n%s\nhere:\n%s\n" % (self.desc, self.matrix, matrix)
        return desc


class CharMatrix(object):

    def __init__(self, diagram):
        if type(diagram) != str:
            raise ValueError

        diagram = cleanup(diagram)

        data = {}
        row = 0
        col = 0
        self.cols = 0
        for c in diagram:
            if c == '\n':
                row += 1
                col = 0
                continue
            data[row, col] = c
            col += 1
            self.cols = max(self.cols, col)
        self.data = data
        self.rows = row
        self.diagram = diagram

        tracks = []
        for row in range(self.rows):
            if self[row, 0] not in ' |':
                tracks.append(row)
        self.rank = len(tracks)
        self.tracks = tracks

    def clone(self):
        return CharMatrix(self.diagram)

    def __getitem__(self, key):
        if type(key) is int:
            key = key, slice(None)
        row, col = key
        return self.data.get((row, col), ' ')

    def __setitem__(self, key, value):
        if type(key) is int:
            key = key, slice(None)
        row, col = key
        self.data[row, col] = value

    def __delitem__(self, key):
        if type(key) is int:
            key = key, slice(None)
        row, col = key
        del self.data[row, col]

    def __str__(self):
        lines = []
        for row in range(self.rows):
            line = []
            for col in range(self.cols):
                line.append(self[row, col])
            line.append('\n')
            lines.append(''.join(line))
        return ''.join(lines)

    def has_connection(self, mask, row, col):
        assert row in self.tracks
        if self[row-1, col] == '|' or self[row+1, col] == '|':
            return True
        return False

    def find_connections(self, mask, row, col):
        assert self.has_connection(mask, row, col)
        assert self[row-1, col] != '|'
        rows = [row]
        row1 = row+1
        while row1 < self.rows:
            op = self[row1, col]
            if op == '|':
                pass
            elif op == '-':
                pass
            elif op == ' ':
                break
            else:
                assert row1 in self.tracks
                rows.append(row1)
            row1 += 1
        return rows

    def build_compound(self, rows, col, namespace):
        tracks = self.tracks
        rank = self.rank
        ops = [self[row, col] for row in rows]
        if '.' in ops:
            ctrl_bits = [tracks.index(rows[idx])
                for idx in range(len(ops)) if ops[idx]=='.']

            A = Qu((2,)*(2*rank), 'ud'*rank)

            for ctrl in genidx((2,)*len(ctrl_bits)):
                factors = [Gate.I] * rank
                for idx, bit_idx in enumerate(ctrl_bits):
                    factors[bit_idx] = Gate.dyads[ctrl[idx]]
                if 0 not in ctrl: # all 1's
                    for row in rows:
                        op = self[row, col]
                        if op != '.':
                            op = 'X' if op == '+' else op
                            G = namespace.get(op)
                            if G is None:
                                raise ParseError(self, row, col, "cannot find gate %r"%op)
                            factors[tracks.index(row)] = G
                A += reduce(matmul, factors)

        elif 'x' in ops:
            if ops != ['x', 'x']:
                raise ParseError(self, rows, col, 'expected exactly two swaps (x)')
            row0, row1 = rows
            factors = [Gate.I]*(self.rank-1)
            factors[row0] = Gate.SWAP
            A = reduce(matmul, factors)
            A.swap2(tracks.index(row0)+1, tracks.index(row1))
        else:
            raise ParseError(self, rows, col, 'no control or swap found')
        return A

    def unpack(self, mask, col, namespace):
        tracks = self.tracks
        row = 0
        while row < self.rows:
            op = self[row, col]
            if op == '|':
                raise ParseError(self, row, col, "dangling wire")
            elif op == ' ':
                if row in tracks:
                    raise ParseError(self, row, col, "short or broken wire")
            elif op == '-':
                if row not in tracks:
                    raise ParseError(self, row, col, "new wire")
            elif self.has_connection(mask, row, col):
                if row not in tracks:
                    raise ParseError(self, row, col, "unexpected gate")
                rows = self.find_connections(mask, row, col)
                A = self.build_compound(rows, col, namespace)
                yield A
                # Skip to the end of this vertical wire:
                row = rows[-1]
            elif op == '+':
                raise ParseError(self, row, col, "uncontrolled plus")
            else:
                if row not in tracks:
                    raise ParseError(self, row, col, "unexpected gate")
                A = namespace.get(op)
                if A is None:
                    raise ParseError(self, row, col, "cannot find gate %r"%op)
                block = [Gate.I if tracks[i]!=row else A for i in range(self.rank)]
                A = reduce(matmul, block)
                yield A
            row += 1

    def parse(self, namespace=Gate.__dict__):
        #print
        #print "="*80
        #print self
        #print
        mask = list(self.data.keys())
        circuit = Gate.I ** self.rank
        for col in range(self.cols):
            for A in self.unpack(mask, col, namespace):
                #if A != (Gate.I**self.rank):
                #    print "PARSE:", A.valence
                #else:
                #    print "PARSE: I**n"
                circuit = A * circuit
        return circuit



def parse(diagram, namespace=Gate.__dict__):
    m = CharMatrix(diagram)
    A = m.parse(namespace)
    return A

def gentracks(rows, cols):
    tracks = ['-'*cols+'\n' if i%2 else ' '*cols+'\n' for i in range(2*rows)]
    return ''.join(tracks)

