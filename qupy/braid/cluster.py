#!/usr/bin/env python3

"""

Attempt to bound performance of arbitrary Anyon decoding.

Here are our dummy fusion rules:

CHARGE * CHARGE -> FUZZY
FUZZY * CHARGE -> FUZZY

"""

import os, sys
import random


import numpy

from qupy.smap import SMap
from qupy.braid.tree import Tree


VACUUM, CHARGE, FUZZY = 0, 1, 2


def sublists(items):
    n = len(items)
    assert n < 20
    i = 0
    while i < 2**n:
        result = []
        j = i
        idx = 0
        while j:
            if j%2:
                result.append(items[idx])
            idx += 1
            j >>= 1
        yield result
        i += 1

assert list(sublists([])) == [[]]
assert list(sublists([0])) == [[], [0]]
assert list(sublists([0, 1, 2])) == [
    [], [0], [1], [0, 1], [2], [0, 2], [1, 2], [0, 1, 2]]


class Lattice(object):

    def __init__(self, l, syndrome=None, errors=[]):

        if syndrome is None:
            syndrome = numpy.zeros((l, l), dtype=numpy.int32)
            syndrome[:] = VACUUM
        else:
            syndrome = numpy.array(syndrome, dtype=numpy.int32)

        nbd = {}
        for i in range(l):
            for j in range(l):
                nbd[i, j] = [
                    ((i+1)%l, j), ((i-1)%l, j),
                    (i, (j+1)%l), (i, (j-1)%l)]
        #print nbd

        self.l = l
        self.nbd = nbd
        self.syndrome = syndrome
        errors = [error.clone() for error in errors]
        self.errors = errors # list of Tree's

    def clone(self):
        lattice = Lattice(self.l, self.syndrome, self.errors)
        return lattice

    def str(self):
        l = self.l
        smap = SMap()

        syndrome = self.syndrome
        for i in range(l):
            for j in range(l):
                charge = syndrome[i, j]
                s = ['.', 'x', '?'][charge]
                smap[i, j] = s
        return str(smap)

    def step(self, verbose=False):

        l = self.l
        syndrome = self.syndrome

        i0 = random.randint(0, l-1)
        j0 = random.randint(0, l-1)

        di, dj = random.choice([(1, 0), (0, 1)])

        c0, c1 = (i0, j0), ((i0+di)%l, (j0+dj)%l)
        for (i, j) in [c0, c1]:
            if syndrome[i, j] == VACUUM:
                syndrome[i, j] = CHARGE
            elif syndrome[i, j] == CHARGE:
                syndrome[i, j] = FUZZY
            # FUZZY stays FUZZY

        errors = self.errors
        for error in errors:
            if c0 in error and c1 in error:
                pass
            elif c0 in error:
                error.add(c0, c1)
            elif c1 in error:
                error.add(c1, c0)
            else:
                continue
            break
        else:
            tree = Tree(c0)
            tree.add(c0, c1)
            errors.append(tree)

        if verbose:
            print(self.str())
            print()

    def apply_error(self, p, verbose=False):
        l = self.l
        e = 2*l**2 # number of edges
        M = numpy.random.poisson(p, e).sum()
        for i in range(M):
            self.step(verbose)

    def do_fuse_tree(self, tree):
        for error in self.errors:
            if tree.intersects(error) and not tree.contains(error):
                return FUZZY
        return VACUUM

    def get_fuzzy(self):
        syndrome = self.syndrome
        i0, i1 = numpy.where(syndrome==FUZZY)
        n = len(i0)
        assert len(i1) == n
        idxs = [(i0[i], i1[i]) for i in range(n)]
        return idxs

    def decode(self, fuzz=0, verbose=False):

        l = self.l
        syndrome = self.syndrome

        if verbose:
            print("lattice.decode: syndrome")
            print(self.str())
            print()
            print("lattice.decode: errors")
            for error in self.errors:
                assert len(error)>=2
                if len(error)>2:
                    print(error.str(l))
                    print()

        idxs0, idxs1 = numpy.where(syndrome==CHARGE)
        #idxs0, idxs1 = numpy.where(syndrome!=VACUUM)

        trees = []
        for idx in range(len(idxs0)):
            i, j = idxs0[idx], idxs1[idx]
            tree = Tree((i, j))
            trees.append(tree)

        #if fuzz and len(idxs0):
        #    idx = random.randint(0, len(idxs0)-1)
        #    i, j = idxs0[idx], idxs1[idx]
        #    tree = Tree((i, j))
        #    trees.append(tree)

        # join clusters using "T_1" separation
        nbd = self.nbd
        j = 0
        while j < len(trees):
            cj = trees[j]
            for k in range(j+1, len(trees)):
                ck = trees[k]
                assert len(ck) == 1 
                for c in cj.sites:
                    if ck.root in nbd[c]:
                        cj.add(c, ck.root) # c <-- ck.root
                        trees.pop(k)
                        break
                else:
                    continue
                break
                # meow :-)
            else:
                j += 1 

        if verbose:
            print("Lattice.decode: clusters")
            print(self.str())

        count = 0
        width = 1
        while trees:

            if verbose:
                print("%d clusters: %s" % (len(trees), [len(c) for c in trees]))
                for c in trees:
                    print(c.str(self.l))
                    print()

            for c1 in trees:
              for c2 in trees:
                #print c1, c2
                if c1 is c2:
                    continue
                assert not c1.intersects(c2), (c1, c2)

            # now we fuse all clusters
            trees1 = []
            for c in trees:
                assert c

                charge = self.do_fuse_tree(c)
                if charge != VACUUM:
                    trees1.append(c)

            trees = trees1
            if verbose and trees:
                print(self.str(), '\n')

            # now we grow clusters and join them

            #width *= 2
            width = 1 # ?
            for tree in trees:
                tree.grow(self.nbd, width)

            j = 0
            while j < len(trees):
                cj = trees[j]
                for k in range(j+1, len(trees)):
                    ck = trees[k]
                    if cj.intersects(ck):
                        trees[j] = cj.join(ck)
                        trees.pop(k)
                        break
                else:
                    j += 1

            for tree in trees:
                radius = tree.get_radius()
                if radius+2 >= l//2:
                    if verbose:
                        print("Lattice.decode: large tree")
                    return False

            count += 1
            assert count < l**2 # !!

        return not trees


def putstr(c):
    sys.stdout.write(str(c))
    sys.stdout.flush()

def main():
    l = argv.get('l', 8)
    p = argv.get('p', 0.05)
    N = argv.get('N', 100)
    max_fuzz = argv.get('max_fuzz', 9)
    verbose = argv.verbose

    _seed = argv.get('seed')
    if _seed is not None:
        random.seed(_seed)
        numpy.random.seed(_seed)

    success = 0.
    for i in range(N):

        lattice = Lattice(l)
        lattice.apply_error(p)

        fuzz = lattice.get_fuzzy()
        #putstr("%s "%len(fuzz))
        if len(fuzz)>max_fuzz:
            result = 0 # FAIL
            continue

#        print "fuzz:", fuzz, ' -- ',
        result = None
        for idxs in sublists(fuzz):
#            print idxs,
            lattice2 = lattice.clone()
            for idx in idxs:
                lattice2.syndrome[idx] = CHARGE
            result = lattice2.decode(verbose=verbose)
            if not result:
#                print 'X',
                break
#        print
        assert result is not None
        success += result

    error = (1. - success/N)
    print()
    print(argv)
    print("error rate = %.6f" % error)

    
from qupy.argv import Argv
argv = Argv()

if __name__ == "__main__":

    if argv.profile:
        import cProfile as profile
        profile.run("main()")

    else:
        main()

