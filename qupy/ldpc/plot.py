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

    elif line.startswith("./css.py "):

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
        assert lhs not in attrs
        attrs[lhs] = rhs


print(rows)

desc = {
    "joschka12" : "[[255,9,6]]",
    "joschka16" : "[[400,16,6]]",
    "joschka20" : "[[625,25,8]]",
    "joschka24" : "[[900,36,10]]",
    "joschka28" : "[[1225,49,10]]",
    "joschka32" : "[[1600,64,10]]",
    "joschka36" : "[[2025,81,12]]",
    "joschka40" : "[[2500,100,12]]",
    "joschka44" : "[[3025,121,14]]",
}

keys = list(rows.keys())
keys.sort()

for key in keys:

    if key == "joschka12":
        continue

    row = rows[key]
    print(row)

    xs = [x for (x, y) in row]
    ys = [y for (x, y) in row]
    pyplot.plot(xs, ys, "x-", label=desc[key])
    #pyplot.semilogy(xs, ys, label=desc[key])

y0, y1 = pyplot.ylim()
pyplot.ylim((y0, 1.))
pyplot.xlabel("bitflip noise rate")
pyplot.ylabel("decoder error rate")

pyplot.legend(loc='upper left')
#pyplot.save("Threshold.pdf")

pyplot.show()



