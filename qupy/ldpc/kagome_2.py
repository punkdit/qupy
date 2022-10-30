#!/usr/bin/env python


import numpy

from qupy.ldpc import solve
from qupy.ldpc.solve import parse, shortstr, array2, zeros2, eq2, compose2, dot2
from qupy.ldpc.solve import intersect
from qupy.ldpc.chain import Chain, Morphism, pushout, equalizer



srcs = []
Hx = parse("""
1........1........1........
1............1.....1.......
..1..............1.......1.
..1..........1..........1..
.1.......1............1....
.1...............1...1.....
........1.......1...1......
........1...1......1.......
.......1...1..............1
.......1........1........1.
......1....1..........1....
......1.....1..........1...
....1.....1.......1........
....1..........1....1......
...1......1...............1
...1..........1.........1..
.....1.........1.....1.....
.....1........1........1...
""")
Hz = parse("""
.1.....1...1.....1....1..1.
111...1111.111..11.........
.1..1....1.....1..1..1.....
..1.....1....1..1..1.....1.
1.11.1....1..11..11..1.....
....1..1..1.....1...1.....1
1..1......1..1....1.....1..
1.1.1111...1.1..111.11.1...
""")
code = Chain([Hx, Hz.transpose()])
srcs.append(code)

Hx = parse("""
..1.............1..1.......
1............1.....1.......
.1.........1.............1.
..1..............1.......1.
.1.......1............1....
1...........1.........1....
.......1..1.........1......
........1.......1...1......
.......1...1..............1
......1.......1...........1
........1......1.......1...
......1.....1..........1...
....1.....1.......1........
...1.........1....1........
.....1...........1......1..
...1..........1.........1..
....1....1...........1.....
.....1.........1.....1.....
""")
Hz = parse("""
111...111..........11.11.11
..1....1...1....1...1....1.
1.1....11..11......11..1.1.
11..11.......1...11..11..1.
1...1....1...1....1...1....
..1....1..1..11..111....1.1
..11.........1...1.1....1..
....1...1.1....1....11.....
""")
code = Chain([Hx, Hz.transpose()])
srcs.append(code)
Hx = parse("""
.......1........1........1.
..1.............1..1.......
.1.........1.............1.
......1....1..........1....
........1...1......1.......
1...........1.........1....
.......1..1.........1......
...1......1...............1
.....1........1........1...
......1.......1...........1
....1..........1....1......
........1......1.......1...
..1..........1..........1..
...1.........1....1........
.1...............1...1.....
.....1...........1......1..
....1....1...........1.....
1........1........1........
""")
Hz = parse("""
.........1.111..1111.11.11.
......11..11.............11
..........111.111..11.11.11
11..........11..1111....11.
.11.............11......11.
...1.1.......11.........1.1
.......11......11..11......
...11....11.......1.1......
""")
code = Chain([Hx, Hz.transpose()])
srcs.append(code)


Hx = parse("""
11.........................1..1.......................1........1.................
.11.........................1..1.......................1........1................
1.1..........................1..1.......................1........1...............
...11.........................1..1.......................1........1..............
....11.........................1..1.......................1........1.............
...1.1..........................1..1.......................1........1............
......11...................1.....1..........................1........1...........
.......11...................1.....1..........................1........1..........
......1.1....................1.....1..........................1........1.........
.........11.........................1..1.......................1........1........
..........11.........................1..1.......................1........1.......
.........1.1..........................1..1.......................1........1......
............11.........................1..1.......................1........1.....
.............11.........................1..1.......................1........1....
............1.1..........................1..1.......................1........1...
...............11...................1.....1..........................1........1..
................11...................1.....1..........................1........1.
...............1.1....................1.....1..........................1........1
..................11.........................1..1.....1.................1........
...................11.........................1..1.....1.................1.......
..................1.1..........................1..1.....1.................1......
.....................11.........................1..1.....1.................1.....
......................11.........................1..1.....1.................1....
.....................1.1..........................1..1.....1.................1...
........................11...................1.....1........1.................1..
.........................11...................1.....1........1.................1.
........................1.1....................1.....1........1.................1
""")
Hz = parse("""
1.....1....................1.1...................................................
.1.....1...................11....................................................
..1.....1...................11...................................................
1..1..........................1.1................................................
.1..1.........................11.................................................
..1..1.........................11................................................
...1..1..........................1.1.............................................
....1..1.........................11..............................................
.....1..1.........................11.............................................
.........1.....1....................1.1..........................................
..........1.....1...................11...........................................
...........1.....1...................11..........................................
.........1..1..........................1.1.......................................
..........1..1.........................11........................................
...........1..1.........................11.......................................
............1..1..........................1.1....................................
.............1..1.........................11.....................................
..............1..1.........................11....................................
..................1.....1....................1.1.................................
...................1.....1...................11..................................
....................1.....1...................11.................................
..................1..1..........................1.1..............................
...................1..1.........................11...............................
....................1..1.........................11..............................
.....................1..1..........................1.1...........................
......................1..1.........................11............................
.......................1..1.........................11...........................
1.................1...................................1.1........................
.1.................1..................................11.........................
..1.................1..................................11........................
...1.................1...................................1.1.....................
....1.................1..................................11......................
.....1.................1..................................11.....................
......1.................1...................................1.1..................
.......1.................1..................................11...................
........1.................1..................................11..................
1........1.....................................................1.1...............
.1........1....................................................11................
..1........1....................................................11...............
...1........1.....................................................1.1............
....1........1....................................................11.............
.....1........1....................................................11............
......1........1.....................................................1.1.........
.......1........1....................................................11..........
........1........1....................................................11.........
.........1........1.....................................................1.1......
..........1........1....................................................11.......
...........1........1....................................................11......
............1........1.....................................................1.1...
.............1........1....................................................11....
..............1........1....................................................11...
...............1........1.....................................................1.1
................1........1....................................................11.
.................1........1....................................................11
...........................1.................1........1.....1....................
............................1.................1........1.....1...................
.............................1.................1........1.....1..................
..............................1.................1.....1..1.......................
...............................1.................1.....1..1......................
................................1.................1.....1..1.....................
.................................1.................1.....1..1....................
..................................1.................1.....1..1...................
...................................1.................1.....1..1..................
...........................1........1..........................1.....1...........
............................1........1..........................1.....1..........
.............................1........1..........................1.....1.........
..............................1........1.......................1..1..............
...............................1........1.......................1..1.............
................................1........1.......................1..1............
.................................1........1.......................1..1...........
..................................1........1.......................1..1..........
...................................1........1.......................1..1.........
....................................1........1..........................1.....1..
.....................................1........1..........................1.....1.
......................................1........1..........................1.....1
.......................................1........1.......................1..1.....
........................................1........1.......................1..1....
.........................................1........1.......................1..1...
..........................................1........1.......................1..1..
...........................................1........1.......................1..1.
............................................1........1.......................1..1
""")


tgt = Chain([Hx, Hz.transpose()])
print("tgt:", tgt)

cmaps = [
[
    parse("""
    1.................
    ..................
    .1................
    ..................
    ..1...............
    ...1..............
    ....1.............
    .....1............
    ..................
    ..................
    ......1...........
    .......1..........
    ........1.........
    .........1........
    ..................
    ..........1.......
    ..................
    ...........1......
    ............1.....
    .............1....
    ..................
    ..............1...
    ..................
    ...............1..
    ..................
    ................1.
    .................1
    """),
    parse("""
    1..........................
    ...........................
    ...........................
    ...........................
    ...........................
    ..1........................
    ...........................
    .1.........................
    ...........................
    ...........................
    ...........................
    ........1..................
    ...........................
    .......1...................
    ...........................
    ......1....................
    ...........................
    ...........................
    ...........................
    ....1......................
    ...........................
    ...1.......................
    ...........................
    ...........................
    ...........................
    ...........................
    .....1.....................
    .........1.................
    ...........................
    ...........................
    ...........................
    ...........................
    .............1.............
    ...........................
    .................1.........
    ...........................
    ...........................
    ...........................
    ............1..............
    ...........................
    ................1..........
    ...........................
    ...........1...............
    ...........................
    ...........................
    ...........................
    ...............1...........
    ...........................
    ..........1................
    ...........................
    ...........................
    ...........................
    ...........................
    ..............1............
    ..................1........
    ...........................
    ...........................
    ...........................
    ...........................
    ........................1..
    ...........................
    .....................1.....
    ...........................
    ...........................
    ...........................
    ...................1.......
    ...........................
    .........................1.
    ...........................
    ......................1....
    ...........................
    ...........................
    ...........................
    ....................1......
    ...........................
    ..........................1
    ...........................
    ...........................
    ...........................
    ...........................
    .......................1...
    """),
    parse("""
    ........
    ........
    ........
    ......1.
    .1.11..1
    .1.11..1
    ........
    11..1..1
    ........
    .1......
    ........
    ........
    ........
    .1.1.1.1
    ........
    ........
    ........
    ........
    ....1..1
    ..1.1..1
    ....1..1
    ........
    ........
    ........
    ....1...
    ........
    ........
    ....1..1
    ....1..1
    ....1..1
    ......1.
    ........
    ........
    ........
    ..1.1..1
    ........
    .1......
    .1.1....
    .1.1....
    ........
    1..1....
    ........
    ........
    ........
    ........
    ........
    .....1.1
    ........
    ........
    ........
    ........
    .......1
    ........
    ........
    ..1....1
    ........
    ........
    ....1.1.
    ........
    ........
    ....1...
    ........
    ........
    .1.....1
    ........
    ........
    .1.1...1
    ........
    ........
    11.....1
    ........
    ........
    .......1
    ........
    ........
    .....1..
    ........
    ........
    ........
    ........
    ........
    """),
],
[
    parse("""
    ..................
    1.................
    .1................
    ..1...............
    ...1..............
    ..................
    ....1.............
    ..................
    .....1............
    ......1...........
    .......1..........
    ..................
    ........1.........
    ..................
    .........1........
    ..................
    ..........1.......
    ...........1......
    ............1.....
    ..................
    .............1....
    ..................
    ..............1...
    ...............1..
    ................1.
    .................1
    ..................
    """),
    parse("""
    ...........................
    ...........................
    ...................1.......
    ...........................
    .........................1.
    ...........................
    ......................1....
    ...........................
    ...........................
    ...........................
    ....................1......
    ...........................
    ..........................1
    ...........................
    ...........................
    ...........................
    ...........................
    .......................1...
    ..................1........
    ...........................
    ...........................
    ...........................
    ...........................
    ........................1..
    ...........................
    .....................1.....
    ...........................
    ...........................
    ...........................
    1..........................
    ...........................
    ..1........................
    ...........................
    .1.........................
    ...........................
    ...........................
    ...........................
    ........1..................
    ...........................
    .......1...................
    ...........................
    ...........................
    ...........................
    ...........................
    ......1....................
    ....1......................
    ...........................
    ...........................
    ...........................
    ...........................
    ...1.......................
    ...........................
    .....1.....................
    ...........................
    ...........................
    ...........................
    .............1.............
    ...........................
    .................1.........
    ...........................
    .........1.................
    ...........................
    ...........................
    ...........................
    ................1..........
    ...........................
    ...........1...............
    ...........................
    ...........................
    ...........................
    ...........................
    ............1..............
    ..........1................
    ...........................
    ...........................
    ...........................
    ...........................
    ..............1............
    ...........................
    ...............1...........
    ...........................
    """),
    parse("""
    1.111...
    ........
    ........
    ........
    111..11.
    ........
    ........
    ........
    ........
    1.1.....
    .......1
    1.1.....
    ........
    ........
    ........
    1.......
    ........
    ........
    ........
    ........
    ........
    .....1..
    .....11.
    .....11.
    ........
    ...1....
    ........
    ...11...
    .....11.
    .....11.
    ........
    ...1.11.
    ........
    ........
    ........
    ........
    1.1.....
    111.....
    1.1.....
    ........
    ........
    ........
    ..1.....
    ........
    ........
    ........
    ........
    ........
    .....1..
    ........
    ........
    ........
    .......1
    ........
    ...11...
    ........
    ........
    .....11.
    ........
    ........
    ...1....
    ........
    ........
    1.1.....
    ........
    ........
    111.....
    ........
    ........
    1.......
    ........
    ........
    .......1
    ........
    ........
    .....1..
    ........
    ........
    ........
    ........
    ........
    """),
],
[
    parse("""
    1.................
    .1................
    ..................
    ..1...............
    ..................
    ...1..............
    ..................
    ....1.............
    .....1............
    ......1...........
    ..................
    .......1..........
    ..................
    ........1.........
    .........1........
    ..........1.......
    ...........1......
    ..................
    ..................
    ............1.....
    .............1....
    ..............1...
    ...............1..
    ..................
    ................1.
    ..................
    .................1
    """),
    parse("""
    ...........................
    ................1..........
    ...........................
    ...........1...............
    ...........................
    ...........................
    ...........................
    ...........................
    ............1..............
    ..........1................
    ...........................
    ...........................
    ...........................
    ...........................
    ..............1............
    ...........................
    ...............1...........
    ...........................
    ...........................
    ...........................
    .............1.............
    ...........................
    .................1.........
    ...........................
    .........1.................
    ...........................
    ...........................
    ...........................
    ...................1.......
    ...........................
    .........................1.
    ...........................
    ...........................
    ...........................
    ...........................
    ......................1....
    ....................1......
    ...........................
    ...........................
    ...........................
    ...........................
    ..........................1
    ...........................
    .......................1...
    ...........................
    ...........................
    ...........................
    ..................1........
    ...........................
    ........................1..
    ...........................
    .....................1.....
    ...........................
    ...........................
    ...........................
    ..1........................
    ...........................
    .1.........................
    ...........................
    ...........................
    ...........................
    ...........................
    1..........................
    .......1...................
    ...........................
    ...........................
    ...........................
    ...........................
    ......1....................
    ...........................
    ........1..................
    ...........................
    ...........................
    ...........................
    ...1.......................
    ...........................
    .....1.....................
    ...........................
    ....1......................
    ...........................
    ...........................
    """),
    parse("""
    1.11....
    ......1.
    1.11....
    ........
    ........
    ........
    1.1.....
    ........
    ........
    ........
    ........
    ........
    .1......
    ..1..1..
    ..1..1..
    ........
    ..1.....
    ........
    1..1...1
    ........
    ........
    ........
    1..111..
    ........
    ........
    ........
    ........
    1..1.1..
    1..111..
    1..1.1..
    ........
    ........
    ........
    ...1....
    ........
    ........
    ..1..1..
    ..1..1..
    ..1..1..
    .1......
    ........
    ........
    ........
    ......1.
    ........
    .....1.1
    ........
    ........
    ........
    .....1..
    ........
    ........
    ........
    ........
    1..1....
    ........
    ........
    1..11...
    ........
    ........
    1.......
    ........
    ........
    ..1...1.
    ........
    ........
    .11.....
    ........
    ........
    ..1.....
    ........
    ........
    .......1
    ........
    ........
    .....1..
    ........
    ........
    ........
    ........
    ........
    """),
]
]

fs = []
for i in range(3):
    cmap = cmaps[i]
    f = Morphism(srcs[i], tgt, cmap)
    fs.append(f)


def main(srcs, tgt, fs):

    for i in [-1, 0, 1, 2]:
        print(len(tgt.homology(i)))
    return

    for src in srcs:
        print(src.homology(0))

    tgt = tgt.get_dual()
    srcs = [src.get_dual() for src in srcs]
    fs = [f.get_dual() for f in fs]
    
    f, g, h = fs
    
    f1, g1, code, _ = pushout(f, g)
    print(code) # Chain(0 <--- 0 <--- 9)

    return
    
    
    Lxs = []
    
    if 1:
        codes = [src.get_code() for src in srcs]
        print()
        
        for code in codes:
            print(code)
            #print(code.distance()) # (3, 6)
        
            LxHx = numpy.concatenate((code.Lx, code.Hx))
            Lxs.append(LxHx)
    
    else:
        for src in srcs:
            Lxs.append(src[0])
    
    
    L01 = intersect(Lxs[0], Lxs[1])
    L02 = intersect(Lxs[0], Lxs[2])
    L12 = intersect(Lxs[1], Lxs[2])
    print(L01.shape)
    print(L02.shape)
    print(L12.shape)
    
    L012 = intersect(L01, Lxs[2])
    print(L012.shape)
    
    print(L012)
    


if __name__ == "__main__":

    main(srcs, tgt, fs)

    print("OK\n")







