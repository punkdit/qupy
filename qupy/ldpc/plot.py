#!/usr/bin/env python3

import math

import numpy
from matplotlib import pyplot

from qupy.argv import argv


def squeeze2(N, p, frac=0.68): # one std dev = 0.68
    "frac of data is bounded by what error bar?"
    data = []
    for i in range(100):
        r = 1.*numpy.random.binomial(N, p) / N
        data.append(r)
    data.sort() # lowest to highest
    #print data
    assert 0.<=frac<=1.
    idx0 = int(0.5*(1-frac)*len(data))
    idx1 = int(0.5*(1+frac)*len(data))
    return p-data[idx0], data[idx1]-p


def __squeeze2(N, p, frac=None):
    assert 0.<=p<=1.
    var = math.sqrt(p*(1.-p)/N)
    print("squeeze2:", N, p, "=", var)
    return max(0., p-var), min(1., p+var)





name = argv.next()

s = open(name).read()

lines = s.split("\n")

rows = {}
attrs = None
for line in lines:

    line = line.strip()
    if not line and not attrs:
        pass

    elif not line:

        key = attrs["code"]
        N = int(attrs["N"])
        p = float(attrs["p"])
        error = float(attrs["error rate"])
        errorbar = squeeze2(N, error)
        #print("errorbar:", errorbar)
        val = (p, error, errorbar)
        row = rows.get(key, [])
        row.append(val)
        rows[key] = row
        attrs = None

    elif line.startswith("#"):
        continue

    elif line.startswith("./main.py "):

        attrs = {}
        for item in line.split():
            if '=' in item:
                lhs, rhs = item.split("=")
                attrs[lhs] = rhs
            if item.endswith(".ldpc"):
                item = item[:-len(".ldpc")]
                if item.startswith("codes/"):
                    item = item[len("codes/"):]
                attrs["code"] = item

        #print(attrs)

    elif " = " in line:
        lhs, rhs = line.split(" = ")
        rhs = float(rhs)
        assert attrs is not None
        assert lhs not in attrs
        attrs[lhs] = rhs


print(rows)

desc = {
#    #"joschka12" : ("MM", "[[255,9,6]]", "x-"),
#    "joschka16"  : ("MM", "[[400,16,6]]", "x-"),
#    #"joschka20" : ("MM", "[[625,25,8]]", "x-"),
#    "joschka24"  : ("MM", "[[900,36,10]]", "x-"),
#    #"joschka28" : ("MM", "[[1225,49,10]]", "x-"),
#    "joschka32"  : ("MM", "[[1600,64,10]]", "x-"),
#    "joschka36"  : ("MM", "[[2025,81,12]]", "x-"),
#    #"joschka40" : ("MM", "[[2500,100,12]]", "x-"),
#    "joschka44"  : ("MM", "[[3025,121,14]]", "x-"),
#    "joschka60"  : ("MM", "[[5625,225,16]]", "x-"),
#    "joschka80"  : ("MM", "[[10000,400,18]]", "x-"),
#    "joschka100" : ("MM", "[[15625,625,20]]", "x-"),
    "qc35"       : ("QC", "[[1954,64,14]]", "x--"),
    "qc45"       : ("QC", "[[3321,81,16]]", "x--"),
    "qc60"       : ("QC", "[[5904,144,20]]", "x--"),
    "qc100_27"   : ("QC", "[[15329,729,24]]", "x--"),
#    "bicycle_500_100" : ("Bi", "[[500,100,12]]", "x-"),
#    "bicycle_1000_200" : ("Bi", "[[1000,200,12]]", "x-"),
#    "bicycle_2000_400" : ("Bi", "[[2000,400,12]]", "x-"),
#    "bicycle_4000_800" : ("Bi", "[[4000,800,12]]", "x-"),
#    "bicycle_8000_1600" : ("Bi", "[[8000,1600,12]]", "x-"),
    #"bicycle_3780_630_12" : ("Bi", "[[3780,630,12]]", "x-"),
    "bicycle_3786_946_12" : ("Bi", "[[3786,946]] rw=12", "x-"),
    "bicycle_3786_946_16" : ("Bi", "[[3786,946]] rw=16", "x-"),
    "bicycle_3786_946_20" : ("Bi", "[[3786,946]] rw=20", "x-"),
    "bicycle_3786_946_24" : ("Bi", "[[3786,946]] rw=24", "x-"),
    "bicycle_3786_946_32" : ("Bi", "[[3786,946]] rw=32", "x-"),
}


def getrank(key): # bit of a hack...
    val = desc.get(key)
    if val is None:
        return ('', 0)
    tp = val[0]
    params = val[1]
    bits = params.split(',')[0]
    assert bits.startswith("[[")
    bits = bits[2:]
    return tp, int(bits)

EPSILON = 1e-8

keys = list(rows.keys())
keys.sort(key = getrank)

if argv.semilogy:
    pyplot.semilogy()

for key in keys:

    if key not in desc:
        continue

    row = rows[key]
    row.sort()
    print(row)

    xs = [x for (x, y, errorbar) in row]
    ys = [y for (x, y, errorbar) in row]
    yerr = [errorbar for (x, y, errorbar) in row]
    yerr = [max(EPSILON, e[0]) for e in yerr], [max(EPSILON, e[1]) for e in yerr]
    fam, label, style = desc[key]
    label = fam+label
    pyplot.errorbar(xs, ys, yerr=yerr, fmt=style, label=label, capsize=4.)

y0, y1 = pyplot.ylim()
pyplot.ylim((y0, 1.))
pyplot.xlabel("bitflip noise rate")
pyplot.ylabel("decoder error rate")

#pyplot.save("Threshold.pdf")

if argv.semilogy:
    pyplot.legend(loc='lower right')
else:
    pyplot.legend(loc='upper left')

pyplot.show()



