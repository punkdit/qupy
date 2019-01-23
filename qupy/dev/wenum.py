#!/usr/bin/env python3

from math import exp, pi
from random import randint, seed

import numpy
from matplotlib import pyplot

from qupy.ldpc.solve import zeros2, dot2, rank, shortstr, enum2, parse
from qupy.ldpc.toric import Toric2D, Toric3D

from qupy.argv import argv


def get_rand_sparse(m, n, weight=4):
    Hz = zeros2(m, n)
    for i in range(m):
        for j in range(weight):
            while 1:
                col = randint(0, n-1)
                if Hz[i, col] == 0:
                    break
            Hz[i, col] = 1
    return Hz


def get_rand_dense(m, n):
    Hz = numpy.random.randint(0, 2, size=(m,n))
    return Hz


def span(Hz):
    assert rank(Hz) == len(Hz)
    m, n = Hz.shape
    print("enum2: %d "%(2**m))
    for v in numpy.ndindex((2,)*m):
        u = dot2(v, Hz)
        yield u


def get_counts_even(Hz):
    n = Hz.shape[1]
    assert n%2==0
    counts = numpy.array([0.] * (n//2 + 1))
    for v in span(Hz):
        i = v.sum()
        #print(i)
        assert i%2 == 0
        counts[i//2] += 1
    counts /= counts.sum()
    return counts


def get_counts(Hz):
    n = Hz.shape[1]
    counts = numpy.array([0.] * (n + 1))
    for v in span(Hz):
        i = v.sum()
        #print(i)
        counts[i] += 1
    #counts /= counts.sum()
    return counts


def sample_counts_even(Hz, trials=10000):
    m, n = Hz.shape
    assert n%2==0
    counts = numpy.array([0.] * (n//2 + 1))
    ri = numpy.random.randint
    for i in range(trials):
        u = ri(0, 2, size=(m,))
        v = dot2(u, Hz)
        i = v.sum()
        #print(i)
        assert i%2 == 0
        counts[i//2] += 1
    counts /= trials
    return counts


def sample_counts(Hz, logop=None, trials=10000):
    m, n = Hz.shape
    counts = numpy.array([0.] * (n + 1))
    if logop is None:
        logop = numpy.zeros((n,), dtype=int)
    ri = numpy.random.randint
    for i in range(trials):
        u = ri(0, 2, size=(m,))
        v = dot2(u, Hz)
        v += logop
        v %= 2
        i = v.sum()
        #print(i)
        counts[i] += 1
    counts /= trials
    return counts


def sample_spin(Hz, logop=None, k=0, trials=10000):
    m, n = Hz.shape
    counts = numpy.array([0., 0])
    if logop is None:
        logop = numpy.zeros((n,), dtype=int)
    ri = numpy.random.randint
    for i in range(trials):
        u = ri(0, 2, size=(m,))
        v = dot2(u, Hz)
        v += logop
        v %= 2
        counts[v[k]] += 1
    counts /= trials
    return counts


def sample_spin_2(Hz, logop=None, k0=0, k1=1, trials=10000):
    m, n = Hz.shape
    counts = numpy.array([0., 0.])
    if logop is None:
        logop = numpy.zeros((n,), dtype=int)
    ri = numpy.random.randint
    for i in range(trials):
        u = ri(0, 2, size=(m,))
        v = dot2(u, Hz)
        v += logop
        v %= 2
        counts[(v[k0]+v[k1])%2] += 1
    counts /= trials
    return counts




def sample_counts_2(Hz, trials=100):
    m, n = Hz.shape
    #counts = numpy.array([0.] * (n + 1))
    counts = {}
    ri = numpy.random.randint
    for i in range(trials):
      for j in range(trials):
        v0 = dot2(ri(0, 2, size=(m,)), Hz)
        v1 = dot2(ri(0, 2, size=(m,)), Hz)
        key = [0, 0, 0, 0]
        for k in range(n):
            key[v0[k] + 2*v1[k]] += 1
        key = tuple(key)
        counts[key] = counts.get(key, 0) + 1
    return counts


def show_poly(counts):
    print(code.Hz)
    print(code.Hz.shape)
    terms = []
    for i, count in enumerate(counts):
        if count==0:
            continue
        terms.append("%d*x**%d" % (count, i))

    s = (" + ".join(terms))
    return s


def get_avg(counts):
    #n = len(counts)
    total = 0
    for i, val in enumerate(counts):
        total += i*val
    return total / counts.sum()


def get_dev(counts):
    avg = get_avg(counts)
    dev = 0
    for i, val in enumerate(counts):
        dev += val * ( (i-avg)**2 )
    dev = dev / counts.sum()
    return dev ** 0.5


def main():

    l = argv.get("l", 4)
    m = argv.get("m", l*l-1)
    n = argv.get("n", 2*l*l)

    weight = argv.get("weight", 4)

    if argv.e8:
        Hz = parse("""
        1....111
        .1..1.11
        ..1.11.1
        ...1111.
        """)
    elif argv.toric:
        code = Toric2D(l)
        Hz = code.Hz
    elif argv.toric3:
        code = Toric3D(l)
        Hz = code.Hx
        print("row weight:", Hz[0].sum())
    elif argv.rand_sparse:
        Hz = get_rand_sparse(m, n, weight)
    elif argv.rand_dense:
        Hz = get_rand_dense(m, n)
    else:
        return

    m, n = Hz.shape

    print("Hz.shape:", (m, n))

    logop = argv.logop
    if logop=="dense":
        logop = get_rand_dense(1, n)
        logop.shape = (n,)
    elif logop=="sparse":
        logop = get_rand_sparse(1, n, weight)
        logop.shape = (n,)

    if argv.spin:
        counts = sample_spin(Hz)
        print(counts)
        return

    if argv.spin_2:
        counts = sample_spin_2(Hz)
        print(counts)
        return

    genus = argv.get("genus", 1)

    if genus==1:
        if argv.exact:
            counts = get_counts(Hz)
        else:
            counts = sample_counts(Hz, logop=logop)
    elif genus==2:

        trials = argv.get("trials", 100)
        counts = sample_counts_2(Hz, trials)

        counts = list(counts.items())
        #counts.sort()
        counts.sort(key = lambda item: -item[1])
        for (k, v) in counts[:40]:
            print(k, v)
        return

    else:
        return

    if argv.show:
        print(list(counts))
    if argv.showpoly:
        cs = list(int(round(x)) for x in counts)
        n = len(cs)
        s = ' + '.join("%d*x**%d*y**%d"%(v, i, n-i-1) 
            for i, v in enumerate(cs) if v)
        print(s)

    avg = get_avg(counts)
    print("get_avg:", avg)

    dev = get_dev(counts)
    print("get_dev:", dev)

    gauss = []
    for i in range(len(counts)):
        x = exp(-(i-avg)**2 / (2*dev**2)) / ((2*pi)**0.5 *dev )
        gauss.append(x)

    if argv.plot:
        pyplot.plot(counts, 'b')
        pyplot.plot(gauss, 'g')
        pyplot.show()

    return

    m, n = Hz.shape

    for i in range(10):
        Hz = get_rand(m, n)
        #print(shortstr(Hz))
        #counts = get_counts(Hz)
        counts = sample_counts(Hz)
        print(repr(counts))

        if argv.plot:
            pyplot.plot(counts, 'g')

    if argv.plot:
        pyplot.show()


def test_poly():
    from qupy.dev.comm import Poly
    x = Poly({(1,0):1})
    y = Poly({(0,1):1})

    # l=3 toric code
    s = "y**18 + 9*x**4 * y**14 + 24*x**6 * y**12 + \
        99*x**8 * y**10 + 72*x**10 * y**8 + 51*x**12 * y**6"

    # l=4 toric code
    s = "1*x**0*y**32 + 16*x**4*y**28 + 32*x**6*y**26 + \
        212*x**8*y**24 + 864*x**10*y**22 + 3344*x**12*y**20 + \
        6784*x**14*y**18 + 10262*x**16*y**16 + 6784*x**18*y**14 + \
        3344*x**20*y**12 + 864*x**22*y**10 + 212*x**24*y**8 + \
        32*x**26*y**6 + 16*x**28*y**4 + 1*x**32*y**0"

    p = eval(s, {"x":x, "y":y})
    print(p)

    p = eval(s, {"x":x+y, "y":x-y})
    p = p/32768.
    print(p)
    


if __name__ == "__main__":

    _seed = argv.seed
    if _seed is not None:
        seed(_seed)
        numpy.random.seed(_seed)

    fn = argv.next() or "main"
    fn = eval(fn)
    fn()


