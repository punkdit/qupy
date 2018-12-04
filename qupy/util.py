#!/usr/bin/env python3

from qupy.scalar import EPSILON


# does not need hashable operators
def mulclose(gen, verbose=False, maxsize=None):
    ops = list(gen)
    bdy = gen
    while bdy:
        _bdy = []
        for g in bdy:
            for h in gen:
                k = g*h
                if k not in ops:
                    ops.append(k)
                    _bdy.append(k)
        bdy = _bdy
        if verbose:
            print("mulclose:", len(ops))
        if maxsize and len(ops) >= maxsize:
            break
    return ops


# uses hashable operators
def mulclose_fast(gen, verbose=False, maxsize=None):
    els = set(gen)
    bdy = list(els)
    changed = True 
    while bdy: 
        #if verbose:
        #    print "mulclose:", len(els)
        _bdy = [] 
        for A in gen: 
            for B in bdy: 
                C = A*B  
                if C not in els: 
                    els.add(C)
                    _bdy.append(C)
                    if maxsize and len(els)>=maxsize:
                        return list(els)
        bdy = _bdy 
    return els  





def show_spec(A):
    items = A.eigs()
    for val, vec in items:
        print(val, vec.shortstr())


class FuzzyDict(object):
    def __init__(self, epsilon=EPSILON):
        self._items = [] 
        self.epsilon = epsilon

    def __getitem__(self, x):
        epsilon = self.epsilon
        for key, value in self._items:
            if abs(key - x) < epsilon:
                return value
        raise KeyError

    def get(self, x, default=None):
        epsilon = self.epsilon
        for key, value in self._items:
            if abs(key - x) < epsilon:
                return value
        return default

    def __setitem__(self, x, y):
        epsilon = self.epsilon
        items = self._items
        idx = 0
        while idx < len(items):
            key, value = items[idx]
            if abs(key - x) < epsilon:
                items[idx] = (x, y)
                return
            idx += 1 
        items.append((x, y))

    def __str__(self):
        return "{%s}"%(', '.join("%s: %s"%item for item in self._items))

    def items(self):
        for key, value in self._items:
            yield (key, value)


