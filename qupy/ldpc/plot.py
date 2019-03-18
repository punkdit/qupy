#!/usr/bin/env python3

from matplotlib import pyplot


from qupy.argv import argv

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
        val = (float(attrs["p"]), float(attrs["error rate"]))
        row = rows.get(key, [])
        row.append(val)
        rows[key] = row
        attrs = None

    elif line.startswith("./main.py "):

        attrs = {}
        for item in line.split():
            if '=' in item:
                lhs, rhs = item.split("=")
                attrs[lhs] = rhs
            if item.endswith(".ldpc"):
                item = item[:-len(".ldpc")]
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
    #"joschka12" : ("MM", "[[255,9,6]]", "x-"),
    "joschka16"  : ("MM", "[[400,16,6]]", "x-"),
    #"joschka20" : ("MM", "[[625,25,8]]", "x-"),
    "joschka24"  : ("MM", "[[900,36,10]]", "x-"),
    #"joschka28" : ("MM", "[[1225,49,10]]", "x-"),
    "joschka32"  : ("MM", "[[1600,64,10]]", "x-"),
    "joschka36"  : ("MM", "[[2025,81,12]]", "x-"),
    #"joschka40" : ("MM", "[[2500,100,12]]", "x-"),
    "joschka44"  : ("MM", "[[3025,121,14]]", "x-"),
    "joschka60"  : ("MM", "[[5625,225,16]]", "x-"),
    "joschka80"  : ("MM", "[[10000,400,18]]", "x-"),
    "joschka100" : ("MM", "[[15625,625,20]]", "x-"),
    "qc35"       : ("QC", "[[1954,64,14]]", "x--"),
    "qc45"       : ("QC", "[[3321,81,16]]", "x--"),
    "qc60"       : ("QC", "[[5904,144,20]]", "x--"),
    "codes/bicycle_500_100" : ("Bi", "[[500,100,<=12]]", "x-"),
    "codes/bicycle_1000_200" : ("Bi", "[[1000,200,?]]", "x-"),
    "codes/bicycle_3780_630_12" : ("Bi", "[[3780,630,<=12]]", "x-"),
    "codes/bicycle_2000_400" : ("Bi", "[[2000,400,?]]", "x-"),
}

keys = list(rows.keys())
keys.sort()

for key in keys:

    if key not in desc:
        continue

    row = rows[key]
    row.sort()
    print(row)

    xs = [x for (x, y) in row]
    ys = [y for (x, y) in row]
    fam, label, style = desc[key]
    label = fam+label
    if argv.semilogy:
        pyplot.semilogy(xs, ys, style, label=label)
    else:
        pyplot.plot(xs, ys, style, label=label)

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



