#!/usr/bin/env python3

from random import randint

import numpy
import numpy.random as ra
from numpy.linalg import lstsq

from qupy.ldpc.css import CSSCode
from qupy.ldpc.solve import shortstr, zeros2, array2, dot2, parse
from qupy.tool import write, choose
from qupy.argv import argv

from qupy.ldpc.decoder import Decoder, RandomDecoder
#from qupy.ldpc.dynamic import Tanner
from qupy.ldpc.bpdecode import RadfordBPDecoder
from qupy.ldpc.cluster import ClusterCSSDecoder
#from qupy.ldpc import ensemble
#from qupy.ldpc import lp


if __name__ == "__main__":
    import os
    datestr = os.popen('date "+%F %H:%M:%S"').read().strip()



def main():

    m = argv.get('m', 6) # constraints (rows)
    n = argv.get('n', 16) # bits (cols)
    j = argv.get('j', 3)  # column weight (left degree)
    k = argv.get('k', 8)  # row weight (constraint weight; right degree)

    #assert 2*m<=n, "um?"

    max_iter = argv.get('max_iter', 200)
    verbose = argv.verbose
    check = argv.get('check', False)

    strop = shortstr

    Lx = None
    Lz = None
    Hx = None
    Tz = None
    Hz = None
    Tx = None
    build = argv.get('build', False)
    logops_only = argv.get("logops_only", True)

    code = None

    if argv.code == 'ldpc':

        H = ldpc(m, n, j, k)
        code = CSSCode(Hz=H, Hx=zeros2(0, n))

    elif argv.code == 'cycle':

        #n = argv.get('n', 20)
        #m = argv.get('m', 18)

        row = str(argv.get('H', '1101'))

        H = []

        for i in range(n):
            h = [0]*n
            for j, c in enumerate(row):
                h[(i+j)%n] = int(c)
            H.append(h)

        delta = n-m
        for i in range(delta):
            H.pop(i*n//delta-i)

#        while len(H) > m:
#            idx = randint(0, len(H)-1)
#            H.pop(idx)

        Hz = array2(H)

        print(shortstr(Hz))

        #print shortstr(solve.row_reduce(Hz))

#        code = CSSCode(Hz=Hz)
#        print
#        print code
#        print code.weightstr()
#
#        decoder = StarDynamicDistance(code)
#        L = decoder.find(verbose=verbose)
#        print shortstr(L)
#        print "distance:", L.sum()
#        return

    elif argv.code == "opticycle":

        code = opticycle(m, n)

        H = Hz = code.Hz

        #return

    elif argv.code == 'tree':

        k = argv.get('k', 5)
        n = 2**k + 1

        code = build_tree(45)
        print(code)

#        graph = Graph()
#        for i in range(H.shape[0]):
#            edge = list(numpy.where(H[i])[0])
#            #print edge,
#            for i0 in edge:
#              for i1 in edge:
#                if i0!=i1:
#                    graph.add_edge(i0, i1)
#
#        leaves = [idx for idx in range(n) if H[:, idx].sum()<=1]
#        print leaves
#
#
#        k = len(leaves)//2
#        print graph.metric(leaves[1], leaves[k])
#        return
#        for i in range(k-1):
#            #i0 = poprand(leaves)
#            #i1 = poprand(leaves)
#            u = zeros2(1, n)
#            u[0, leaves[i]] = 1
#            u[0, leaves[i+k]] = 1
#            # what about picking three leaves?
#            print (leaves[i], leaves[i+k]),
#            H = append2(H, u)
#        print
#
#        # if we take all the leaves we get degenerate...
#        leaves = [idx for idx in range(n) if H[:, idx].sum()<=1]
#        print "leaves:", leaves
#
#        #print shortstr(H)
#        #print
#        #print shortstr(solve.row_reduce(H))

#        code = CSSCode(Hz=H)
#        print code
#        print code.weightstr()

#        print numpy.where(code.Lx[3])
#
#        return

    elif argv.code == 'bicycle':

        from qupy.ldpc.bicycle import Bicycle_LDPC
        print("Bicycle_LDPC(%s, %s, %s, %s)"%(m, n, j, k))
        b = Bicycle_LDPC(m, n, j, k)
        Hx = Hz = b.H

    elif argv.code == 'toric':

        from qupy.ldpc.toric import Toric2D
        l = argv.get('l', 8)

        toric = Toric2D(l)
        Hx, Hz = toric.Hx, toric.Hz
        strop = toric.strop
        #print("Hx:")
        #print(shortstr(Hx))
        #print("Lx:")
        #print(shortstr(toric.Lx))

    elif argv.code == 'torichie':

        from qupy.ldpc.toric import Toric2DHie
        l = argv.get('l', 8)

        toric = Toric2DHie(l)
        Hx, Hz = toric.Hx, toric.Hz

    elif argv.code == 'surface':

        from qupy.ldpc.toric import Surface
        l = argv.get('l', 8)

        surface = Surface(l)
        #Hx, Hz = surface.Hx, surface.Hz
        #strop = surface.strop
        #print shortstr(Hx)
        code = surface.get_code()

    elif argv.code == 'toric3':

        from qupy.ldpc.toric import Toric3D
        l = argv.get('l', 8)

        toric = Toric3D(l)
        Hx, Hz = toric.Hx, toric.Hz
        strop = toric.strop
        #print shortstr(Hx)

    elif argv.code == 'gcolor':

        from qupy.ldpc.gcolor import  Lattice
        l = argv.get('l', 1)
        lattice = Lattice(l)
        code = lattice.build_code()

        #print "Gx:", code.Gx.shape
        #print shortstr(code.Gx)
        print("Gx rank:", solve.rank(code.Gx))
        print("Hx:")
        print(shortstr(code.Hx))

    elif argv.code == 'self_ldpc':

        from qupy.ldpc.search import SelfContainingLDPC_Code
        ldpc_code = SelfContainingLDPC_Code(m, n, j, k)
        ldpc_code.search(max_step=argv.get('max_step', None), verbose=verbose)
        #print ldpc_code

        H = ldpc_code.H
        print("H:")
        print(shortstr(H))
        H = solve.linear_independent(H)
        Hx = Hz = H

        code = CSSCode(Hx=Hx, Hz=Hz)

    elif argv.code == 'cssldpc':

        from qupy.ldpc.search import CSS_LDPC_Code
        ldpc_code = CSS_LDPC_Code(m, n, j, k)
        ldpc_code.search(max_step=argv.get('max_step', None), verbose=verbose)
        #print ldpc_code

        Hx, Hz = ldpc_code.Hx, ldpc_code.Hz
        #print "Hx:"
        #print shortstr(Hx)
        Hx = solve.linear_independent(Hx)
        Hz = solve.linear_independent(Hz)
        #Hx = Hz = H

        code = CSSCode(Hx=Hx, Hz=Hz)

    elif argv.code == "hpack":

        code = hpack(n, j=j, k=k, check=check, verbose=verbose)

    elif argv.code == 'randldpc':

        mz = argv.get('mz', n//3)
        rw = argv.get('rw', 12) # row weight
        code = randldpc(n, mz, rw, check=check, verbose=verbose)

    elif argv.code == 'randcss':

        mx = argv.get('mx', n//3)
        mz = argv.get('mz', n//3)
        distance = argv.get("code_distance")
        code = randcss(n, mx, mz, distance=distance, 
            check=check, verbose=verbose)

    elif argv.code == 'sparsecss':

        mx = argv.get('mx', n//3)
        mz = argv.get('mz', n//3)
        weight = argv.get('weight', 4)
        code = sparsecss(n, mx, mz, weight, check=check, verbose=verbose)

    elif argv.code == 'ensemble':

        mx = argv.get('mx', n//3)
        mz = argv.get('mz', mx)
        rw = argv.get('rw', 12) # row weight
        maxw = argv.get('maxw', 4*rw)
        C = argv.get("C", 100)
        code = ensemble.build(n, mx, mz, rw, maxw, C,
            check=check, verbose=verbose)

    elif argv.code == "rm" or argv.code == "reed_muller":
        from qupy.ldpc import reed_muller
        r = argv.get("r", 1)
        m = argv.get("m", 4)
        puncture = argv.puncture
        cl_code = reed_muller.build(r, m, puncture)
        G = cl_code.G
        G = array2([row for row in G if row.sum()%2==0])
        code = CSSCode(Hx=G, Hz=G)

    elif argv.code == "qrm":
        from qupy.ldpc.gallagher import get_code, hypergraph_product
        from qupy.ldpc import reed_muller
        r = argv.get("r", 1)
        m = argv.get("m", 4)
        puncture = argv.puncture
        cl_code = reed_muller.build(r, m, puncture)
        H = cl_code.G
        H = array2([row for row in H if row.sum()%2==0])
        print(H)
        m, n = 5, 5
        J = zeros2(m, n)
        for i in range(m):
            J[i, i] = 1
            J[i, (i+1)%n] = 1
        print(J)
        Hx, Hz, Lx, Lz = hypergraph_product(H, J)
        code = CSSCode(Hx=Hx, Hz=Hz, Lx=Lx, Lz=Lz)
        print(code.weightstr())
        #print(shortstr(code.Hz))

    elif argv.code == "qr7":
        H = parse("""
        ...1111
        .11..11
        1.1.1.1
        """)
        code = CSSCode(Hx=H, Hz=H)

    elif argv.code == "golay" or argv.code == "qr23":
        H = parse("""
        1.1..1..11111..........
        1111.11.1....1.........
        .1111.11.1....1........
        ..1111.11.1....1.......
        ...1111.11.1....1......
        1.1.1.111..1.....1.....
        1111...1..11......1....
        11.111...11........1...
        .11.111...11........1..
        1..1..11111..........1.
        .1..1..11111..........1
        """)
        code = CSSCode(Hx=H, Hz=H)

    elif argv.code == "qr31":
        H = parse("""
        1..1..1.1...11.11..............
        11.11.1111..1.11.1.............
        11111111.11.1.....1............
        .11111111.11.1.....1...........
        ..11111111.11.1.....1..........
        ...11111111.11.1.....1.........
        1..111.1.1111.11......1........
        11.111....11...........1.......
        .11.111....11...........1......
        ..11.111....11...........1.....
        ...11.111....11...........1....
        ....11.111....11...........1...
        1..1.1...11.11..............1..
        .1..1.1...11.11..............1.
        ..1..1.1...11.11..............1
        """)
        code = CSSCode(Hx=H, Hz=H)

    elif argv.code == "qr17":
        # not self-dual
        Hz = parse("""
        .11.1...11...1.11
        1.11.1...11...1.1
        11.11.1...11...1.
        .11.11.1...11...1
        1.11.11.1...11...
        .1.11.11.1...11..
        ..1.11.11.1...11.
        ...1.11.11.1...11
        1...1.11.11.1...1
        11...1.11.11.1...
        .11...1.11.11.1..
        ..11...1.11.11.1.
        ...11...1.11.11.1
        1...11...1.11.11.
        .1...11...1.11.11
        1.1...11...1.11.1
        11.1...11...1.11.
        """)
        Hx = parse("""
        ...1.111..111.1..
        11.1.....1.111..1
        111..111.1.....1.
        ....1.111..111.1.
        111.1.....1.111..
        .111..111.1.....1
        .....1.111..111.1
        .111.1.....1.111.
        1.111..111.1.....
        1.....1.111..111.
        ..111.1.....1.111
        .1.111..111.1....
        .1.....1.111..111
        1..111.1.....1.11
        ..1.111..111.1...
        1.1.....1.111..11
        11..111.1.....1.1
        """)
        Hz = solve.linear_independent(Hz)
        Hx = solve.linear_independent(Hx)
        code = CSSCode(Hx=Hx, Hz=Hz)

    elif argv.code == "qr41":
        # not self-dual
        Hz = parse("""
        .11.11..111.....1.1.11.1.1.....111..11.11
        1.11.11..111.....1.1.11.1.1.....111..11.1
        11.11.11..111.....1.1.11.1.1.....111..11.
        .11.11.11..111.....1.1.11.1.1.....111..11
        1.11.11.11..111.....1.1.11.1.1.....111..1
        11.11.11.11..111.....1.1.11.1.1.....111..
        .11.11.11.11..111.....1.1.11.1.1.....111.
        ..11.11.11.11..111.....1.1.11.1.1.....111
        1..11.11.11.11..111.....1.1.11.1.1.....11
        11..11.11.11.11..111.....1.1.11.1.1.....1
        111..11.11.11.11..111.....1.1.11.1.1.....
        .111..11.11.11.11..111.....1.1.11.1.1....
        ..111..11.11.11.11..111.....1.1.11.1.1...
        ...111..11.11.11.11..111.....1.1.11.1.1..
        ....111..11.11.11.11..111.....1.1.11.1.1.
        .....111..11.11.11.11..111.....1.1.11.1.1
        1.....111..11.11.11.11..111.....1.1.11.1.
        .1.....111..11.11.11.11..111.....1.1.11.1
        1.1.....111..11.11.11.11..111.....1.1.11.
        .1.1.....111..11.11.11.11..111.....1.1.11
        """)
        Hx = parse("""
        ...1..11...11111.1.1..1.1.11111...11..1..
        1111...11..1.....1..11...11111.1.1..1.1.1
        111.1.1..1.1.11111...11..1.....1..11...11
        ....1..11...11111.1.1..1.1.11111...11..1.
        11111...11..1.....1..11...11111.1.1..1.1.
        1111.1.1..1.1.11111...11..1.....1..11...1
        .....1..11...11111.1.1..1.1.11111...11..1
        .11111...11..1.....1..11...11111.1.1..1.1
        11111.1.1..1.1.11111...11..1.....1..11...
        1.....1..11...11111.1.1..1.1.11111...11..
        1.11111...11..1.....1..11...11111.1.1..1.
        .11111.1.1..1.1.11111...11..1.....1..11..
        .1.....1..11...11111.1.1..1.1.11111...11.
        .1.11111...11..1.....1..11...11111.1.1..1
        ..11111.1.1..1.1.11111...11..1.....1..11.
        ..1.....1..11...11111.1.1..1.1.11111...11
        1.1.11111...11..1.....1..11...11111.1.1..
        ...11111.1.1..1.1.11111...11..1.....1..11
        1..1.....1..11...11111.1.1..1.1.11111...1
        .1.1.11111...11..1.....1..11...11111.1.1.
        """)
        code = CSSCode(Hz=Hz, Hx=Hx)

    elif argv.code == "qr47":
        H = parse("""
        1...11..11.11..1..1.1..11......................
        11..1.1.1.11.1.11.1111.1.1.....................
        111.1..11.....111111.111..1....................
        11111......11...11.1..1....1...................
        .11111......11...11.1..1....1..................
        1.11..1.11.11111...111.1.....1.................
        11.1.1.11.11.11.1.1..111......1................
        111..11.......1..1111.1........1...............
        .111..11.......1..1111.1........1..............
        1.11.1.1.1.11..11.11.111.........1.............
        11.1.11..111.1.11111..1...........1............
        .11.1.11..111.1.11111..1...........1...........
        1.111..1.1...1...1.1.1.1............1..........
        11.1.....1111.11......11.............1.........
        111..1..111..1..1.1.1.................1........
        .111..1..111..1..1.1.1.................1.......
        ..111..1..111..1..1.1.1.................1......
        ...111..1..111..1..1.1.1.................1.....
        1.....1.1..1.111.11...11..................1....
        11..11.11..1..1.1..11......................1...
        .11..11.11..1..1.1..11......................1..
        ..11..11.11..1..1.1..11......................1.
        ...11..11.11..1..1.1..11......................1
        """)
        code = CSSCode(Hx=H, Hz=H)

    elif argv.code == "qr71":
        H = parse("""
        1.1.1.11.1...11..11.....1....1...1111..................................
        1111111.111..1.1.1.1....11...11..1...1.................................
        .1111111.111..1.1.1.1....11...11..1...1................................
        ..1111111.111..1.1.1.1....11...11..1...1...............................
        1.11.1..1..11.1.11..1.1.1..111..1.11....1..............................
        1111...1....1.11.....1.111..1.1...1......1.............................
        .1111...1....1.11.....1.111..1.1...1......1............................
        1..1.111.....1..1.1....11111.11.1111.......1...........................
        111.....11...1....11.....1111111............1..........................
        .111.....11...1....11.....1111111............1.........................
        ..111.....11...1....11.....1111111............1........................
        ...111.....11...1....11.....1111111............1.......................
        ....111.....11...1....11.....1111111............1......................
        1.1.11...1.......1.....1.....1111................1.....................
        .1.1.11...1.......1.....1.....1111................1....................
        ..1.1.11...1.......1.....1.....1111................1...................
        ...1.1.11...1.......1.....1.....1111................1..................
        1.1....11.....1..11..1..1..1.1.......................1.................
        .1.1....11.....1..11..1..1..1.1.......................1................
        ..1.1....11.....1..11..1..1..1.1.......................1...............
        ...1.1....11.....1..11..1..1..1.1.......................1..............
        ....1.1....11.....1..11..1..1..1.1.......................1.............
        .....1.1....11.....1..11..1..1..1.1.......................1............
        ......1.1....11.....1..11..1..1..1.1.......................1...........
        1.1.1.1......1.1.11..1...1..11.1.1.1........................1..........
        1111111..1...1..11.1..1.1.1...1.11.1.........................1.........
        11.1.1...11..1......1..111.1.1.1...1..........................1........
        11.....1.111.1...11..1...11.111.1111...........................1.......
        11..1.11111111...1.1..1.1.11..11................................1......
        .11..1.11111111...1.1..1.1.11..11................................1.....
        ..11..1.11111111...1.1..1.1.11..11................................1....
        ...11..1.11111111...1.1..1.1.11..11................................1...
        ....11..1.11111111...1.1..1.1.11..11................................1..
        1.1.11.1...11..11.....1....1...1111..................................1.
        .1.1.11.1...11..11.....1....1...1111..................................1
        """)
        code = CSSCode(Hx=H, Hz=H)

    elif argv.code == "rm_2_5_p":
        H = parse("""
        ...11111111....................
        .11..1111..11..................
        1.1.1.11.1.1.1.................
        11.1..1.11.1..1................
        ...1111........1111............
        .11..11........11..11..........
        1.1.1.1........1.1.1.1.........
        11.1..1.........11.1..1........
        .11111111......11......11......
        1.111111.1.....1.1.....1.1.....
        11.1111.11......11.....1..1....
        111.1111...1...1...1...1...1...
        1111.11.1..1....1..1...1....1..
        11111.1..1.1.....1.1...1.....1.
        111111.111.1...111.1...1......1
        """)
        code = CSSCode(Hx=H, Hz=H)

    elif argv.code == "to_16_6": # triorthogonal
        H = parse("""
        ........11111111
        ....1111....1111
        11111111........
        ..11..11..11..11
        .1.1.1.1.1.1.1.1
        1..1.11..11.1..1
        """) # distance = 4
        code = CSSCode(Hx=H, Hz=H)

    elif argv.code == "randselfdual":
        m = argv.get("m", 4)
        n = argv.get("n", 8)
        rw = argv.get("rw", 4)
        code = randselfdual(m, n, rw)

    else:
        from qupy.ldpc.gallagher import get_code
        code = get_code(argv.code)


    if argv.truncate:
        idx = argv.truncate
        Hx = Hx[:idx]
        Hz = Hz[:idx]

    for arg in argv:
        if arg.endswith('.ldpc'):
            code = CSSCode.load(arg, build=build, check=check,
                rebuild=argv.rebuild)
    
            if argv.rebuild:
                code.save(arg)

            break

    if code is None:
        code = CSSCode(Lx, Lz, Hx, Tz, Hz, Tx,
            build=build, check=check, verbose=verbose, logops_only=logops_only)

    if argv.dual:
        print("dual code...")
        code = code.dual(build=build)

    # take several copies of the code..
    mul = argv.get("mul", 1)
    code = mul * code

    if argv.classical_product:
        H1 = code.Hx
        H2 = H1.transpose()
        Hx, Hz = hypergraph_product(H1, H2)
        code = CSSCode(Hx=Hx, Hz=Hz)

    if argv.product:

        codes = code.product(code.dual())

        for _code in codes:
            print(_code)
            if _code.k:
                _code.save(stem="prod")

    if argv.verbose != 0:
        print(code)
        #print code.longstr()
        #print code.weightstr()
        print(code.weightsummary())

    if argv.shorten:
        code.shorten()
        print(code)
        print(code.weightsummary())

    if argv.prune:
        print("pruning:", argv.prune)
        code.prune_xlogops(argv.prune)
        print(code)
        print(code.weightsummary())

    if argv.split:

        for i in range(4):
            code = x_split(code, build=False)
            print(code)
            print(code.weightsummary())
            code = code.dual()
            code = x_split(code, build=False)
            code = code.dual()
            print(code)
            print(code.weightsummary())
            code.save(stem="split")

    if argv.save:
        code.save(argv.save)

    if argv.symplectic:
        i = randint(0, code.mz-1)
        j = randint(0, code.k-1)
        code.Lz[j] += code.Hz[i]
        code.Lz[j] %= 2
        code.Tx[i] -= code.Lx[j]
        code.Tx[i] %= 2
        code.do_check()
    
        return

    if argv.showcode:
        print(code.longstr())

    if argv.todot:
        todot(code.Hz)
        return

    if argv.tanner:
        Hx = code.Hx
        graph = Tanner(Hx)
        #graph.add_dependent()

        depth = 3

        graphs = [graph]
        for i in range(depth):
            _graphs = []
            for g in graphs:
                left, right = g.split()
                _graphs.append(left)
                _graphs.append(right)
            graphs = _graphs
        print("split", graphs)

        k = code.Lx.shape[0]
        op = code.Lx[1]
        #op = (op + code.Tx[20])%2
        print(strop(op), op.sum())
        print()
        for i in range(Hx.shape[0]):
            if random()<=0.5:
                op = (op+Hx[i])%2
        #print shortstr(op), op.sum()

        print(strop(op), op.sum())
        print()

        #for graph in [left, right]:
        for graph in graphs*2:
        
            #op = graph.minimize(op, verbose=True)
            op = graph.localmin(op, verbose=True)
            print(strop(op), op.sum())
            print()

        #op = graph.localmin(op)
        #print strop(op), op.sum()
        #print
        #print shortstr(op), op.sum()
            
        return

    if argv.distance=='star':
        decoder = StarDynamicDistance(code)
        L = decoder.find(verbose=verbose)
        print(shortstr(L))
        print("distance <=", L.sum())
    elif argv.distance=='stab':
        d = random_distance_stab(code)
        print("distance <=", d)
    elif argv.distance=='pair':
        d = pair_distance(code)
        print("distance <=", d)
    elif argv.distance=='free':
        d = free_distance(code)
        print("distance <=", d)
    elif argv.distance=='lookup':
        d = lookup_distance(code)
        if d <= 4:
            print("distance =", d)
        else:
            print("distance <=", d)

    if type(argv.distance)==str:
        return

    if argv.get('exec'):

        exec(argv.get('exec'))

    N = argv.get('N', 0)
    p = argv.get('p', 0.01)
    weight = argv.weight

    #assert code.Tx is not None, "use code.build ?"

    decoder = get_decoder(argv, argv.decode, code)
    if decoder is None:
        return

    if argv.whack:
        whack(code, decoder, p, N, 
            argv.get("C0", 10), argv.get("error_rate", 0.05), argv.get("mC", 1.2),
            verbose, argv)
        return

    if argv.noerr:
        print("redirecting stderr to stderr.out")
        fd = os.open("stderr.out", os.O_CREAT|os.O_WRONLY)
        os.dup2(fd, 2)

    decoder.strop = strop
    print("decoder:", decoder.__class__.__name__)

    n_ranks = numpy.zeros((code.n,), dtype=numpy.float64)
    n_record = []
    k_ranks = numpy.zeros((code.k,), dtype=numpy.float64)
    k_record = []

    failures = open(argv.savefail, 'w') if argv.savefail else None

    if argv.loadfail:
        loadfail = open(argv.loadfail)
        errs = loadfail.readlines()
        N = len(errs)

    newHx = []
    newHz = []

    distance = code.n
    count = 0
    failcount = 0
    nonuniq = 0

    if argv.weight1:
        # run through all weight=1 errors
        N = code.n

    if argv.weight2:
        # run through all weight=1 errors
        N = code.n**2

    for i in range(N):

        # We use Hz to look at X type errors (bitflip errors)

        if argv.loadfail:
            err_op = parse(errs[i])
            err_op.shape = (code.n,)
        elif argv.weight1:
            err_op = zeros2(code.n)
            err_op[i%code.n] = 1
        elif argv.weight2:
            err_op = zeros2(code.n)
            ai = i%code.n
            bi = (i//code.n)
            err_op[ai] = 1
            err_op[bi] = 1

        elif weight is not None:
            err_op = zeros2(code.n)
            r = 0
            while r < weight:
                idx = randint(0, code.n-1)
                if err_op[idx] == 0:
                    err_op[idx] = 1
                    r += 1
            #print err_op
        else:
            err_op = ra.binomial(1, p, (code.n,))
            err_op = err_op.astype(numpy.int32)

#        err_op = parse("..............................")
#        err_op.shape = (err_op.shape[1],)
        write(str(err_op.sum()))
        #print err_op.shape

        s = dot2(code.Hz, err_op)
        write(":s%d:"%s.sum())

        op = decoder.decode(p, err_op, verbose=verbose, argv=argv)

        c = 'F'
        success = False
        if op is not None:
            op = (op+err_op)%2
            # Should be a codeword of Hz (kernel of Hz)
            if dot2(code.Hz, op).sum() != 0:
                print(dot2(code.Hz, op))
                print("\n!!!!!  BUGBUG  !!!!!", sparsestr(err_op))
                continue
            write("%d:"%op.sum())

            #print "is_stab:"
            #print strop(op)
            # Are we in the image of Hx ? If so, then success.
            #success = code.is_stab(op)
            success = dot2(code.Lz, op).sum()==0

            if success and op.sum():
                nonuniq += 1
            #    print "\n", shortstr(err_op)
            #    return

            c = '.' if success else 'x'

            if argv.k_rank:

                # XX this does not work very well...

                a = dot2(code.Lz, op)
                assert (a.sum()==0) == success

                if not success:
                    #r = -1. // (a.sum()**0.5) # sqrt ??
                    r = -1. / a.sum()
                    k_ranks += r*a
                else:
                    #r = 1. / (code.k**0.5)
                    r = 1. / code.k
                    k_ranks += r*a

                k_record.append((a, success))


            if op.sum() and not success:
                distance = min(distance, op.sum())

                if argv.minop and argv.minop >= op.sum():
                    a = dot2(code.Lz, op)
                    print()
                    print(sparsestr(a))

            if not success and argv.addstabs and op.sum()<=argv.addstabs:
                #write("%d"%op.sum())
                print((sparsestr(op)))
                newHx.append(op)

                A = numpy.concatenate((code.Lx, code.Hx))
                c = solve.solve(A.transpose(), op.transpose()).transpose()
                cL = c[:code.k]
                zstab = dot2(cL, code.Lz)
                print(shortstr(zstab))
                graph = Tanner(code.Hz)
                zstab = graph.minimize(zstab, target=8, verbose=True)
                print(shortstr(zstab))
                newHz.append(zstab)
                break

#                Hz, Hx = code.Hz, append2(code.Hx, op)
#                #if len(Hz) < len(Hx):
#                #    Hx, Hz = Hz, Hx
#                print Hx.shape, Hz.shape
#                code = CSSCode(Hx=Hx, Hz=Hz)
#                if len(Hz) < len(Hx):
#                    code = code.dual()
#                    decoder = get_decoder(argv, argv.decode, code)
#                print
#                print code
#                print code.weightsummary()
#                code.save("addstabs_%d_%d_%d.ldpc"%(code.n, code.k, code.distance))

            if not success and argv.showfail:
                print()
                print("FAIL:")
                print(shortstr(err_op))

        else:
            failcount += 1

            if failures:
                print(shortstr(err_op), file=failures)
                failures.flush()

        if argv.n_rank:
            r = 2.*int(success) - 1
            #print "r =", r
            if err_op.sum():
                r /= err_op.sum()
                #print r
                #print r*err_op
                n_ranks += r*err_op
                n_record.append((err_op, success))

        write(c+' ')
        count += success

    if hasattr(decoder, 'fini'):
        decoder.fini()

    if N:
        print()
        print(datestr)
        print(argv)
        print("error rate = %.8f" % (1. - 1.*count / (i+1)))
        print("fail rate = %.8f" % (1.*failcount / (i+1)))
        print("nonuniq = %d" % nonuniq)
        print("distance <= %d" % distance)

    if argv.n_rank:
        #n_ranks //= N
        n_ranks = list(enumerate(n_ranks))
        n_ranks.sort(key = lambda item : -item[1])
        #print [r[0] for r in n_ranks]
        print(' '.join('%d:%.2f'%r for r in n_ranks))
        #print min(r[1] for r in n_ranks)
        #print max(r[1] for r in n_ranks)
        A = numpy.array([rec[0] for rec in n_record])
        b = numpy.array([rec[1] for rec in n_record])
        print(A.shape, b.shape)
        sol = lstsq(A, b)
        x, res, rk, s = sol
        #print x
        for i, val in enumerate(x):
            if val<0:
                print(i, end=' ')
        print()

    if argv.k_rank:
        #k_ranks //= N
        k_ranks = list(enumerate(k_ranks))
        k_ranks.sort(key = lambda item : -item[1])
        #print [r[0] for r in k_ranks]
        print(' '.join('%d:%.2f'%r for r in k_ranks))
        #print ' '.join(str(r) for r in k_ranks)
        #print min(r[1] for r in k_ranks)
        #print max(r[1] for r in k_ranks)
        A = numpy.array([rec[0] for rec in k_record])
        b = numpy.array([rec[1] for rec in k_record])
        #print A.shape, b.shape
        sol = lstsq(A, b)
        x, res, rk, s = sol
        #print x
        xs = list(enumerate(x))
        xs.sort(key = lambda item : item[1])
        print(' '.join('%d:%.2f'%item for item in xs))

    if argv.addstabs and newHz:
        Hz = append2(code.Hz, newHz[0])
        code = CSSCode(Hx=code.Hx, Hz=Hz)
        code.save(stem="addstabs")
        print(code)
        print(code.weightsummary())

    elif argv.addstabs and newHx: # and newHx.shape[0]>code.Hx.shape[0]:
        newHx = array2(newHx)
        newHx = numpy.concatenate((code.Hx, newHx))
        #print shortstr(newHx)
        #print
        Hx = solve.linear_independent(newHx)
        #print shortstr(Hx)
        #print
        code = CSSCode(Hx=Hx, Hz=code.Hz)
        code.save(stem="addstabs")
        print(code)
        print(code.weightsummary())
    

class MultiDecoder(object):
    def __init__(self, decode0, decode1):
        self.decode0 = decode0
        self.decode1 = decode1

    def decode(self, p, err_op, **kw):
        decode0 = self.decode0
        decode1 = self.decode1
        op = decode0.decode(p, err_op, **kw)
        if op is None:
            write("F")
            op = decode1.decode(p, err_op, **kw)
        return op



class LookupDecoder(object):
    """
        Keep a lookup table of syndromes for all possible errors
        up to a given weight. Use this to decode.
        This is a minimum weight decoder, when it succeeds.
    """
    def __init__(self, code, weight=4, maxsize=256*1024):
        lookup = {}
        n = code.n
        Hz = code.Hz
        Lz = code.Lz
        err = zeros2(n)
        write("LookupDecoder: building...")
        w = 0
        #for w in range(0, weight+1):
        count = 0
        remove = set()
        while w < weight+1 or len(lookup)<maxsize:
            for idxs in choose(list(range(n)), w):
                err[:] = 0
                err[idxs] = 1
                syn = dot2(Hz, err)
                key = syn.tostring()
                other = lookup.get(key)
                if other is None:
                    lookup[key] = idxs
                elif len(other)==w:
                    for idx in other:
                        err[idx] = (err[idx]+1)%2
                    assert dot2(Hz, err).sum()==0
                    #assert dot2(Lz, err).sum()==0, "not an exact decoder."
                    if dot2(Lz, err).sum():
                        remove.add(key)
            if len(lookup)==count:
                break
            count = len(lookup)
            w += 1
        write("\n")
        for key in remove:
            del lookup[key]
        print("lookup size:", len(lookup))
        self.lookup = lookup
        self.code = code

    def decode(self, p, err, **kw):
        code = self.code
        Hz = code.Hz
        syn = dot2(Hz, err)
        #if err.sum() <=
        a = syn.tostring()
        idxs = self.lookup.get(a)
        if idxs is None:
            #write("?")
            return None
        op = zeros2(code.n)
        op[idxs] = 1
        return op



def get_decoder(argv, decode, code):

    d = 2

    decoder = None

    if decode is None:
        pass

    elif '+' in decode:

        items = decode.split('+')
        a = items[0]
        b = '+'.join(items[1:])
        a = get_decoder(argv, a, code)
        b = get_decoder(argv, b, code)
        decoder = MultiDecoder(a, b)

    elif decode=='bp':

        #argv.noerr = True # supress output of RadfordBPDecoder

        decoder = RadfordBPDecoder(d, code.Hz)
        #decoder = PyCodeBPDecoder(d, code.Hz)

    elif decode=='random':
        decoder = RandomDecoder(code)

    elif decode=='metro':
        from qupy.ldpc.decoder import MetroDecoder
        decoder = MetroDecoder(code)

    elif decode=='metrologop':
        from qupy.ldpc.decoder import MetroLogopDecoder
        decoder = MetroDecoder(code)

    elif decode=='star':
        from qupy.ldpc.decoder import StarMetroDecoder
        decoder = StarMetroDecoder(code)

    elif decode=='tempered':
        from qupy.ldpc.decoder import TemperedDecoder
        decoder = TemperedDecoder(code)

    elif decode=='cluster':
        decoder = ClusterCSSDecoder(2, code.Hx, code.Hz)

    elif decode=='mincluster':
        decoder = ClusterCSSDecoder(2, code.Hx, code.Hz)
        decoder.minimize = True

    elif decode=='pma':
        decoder = PMADecoder(code)

    elif decode=='dynamic':
        decoder = DynamicDecoder(code)

    elif decode=='stardynamic':
        decoder = StarDynamicDecoder(code)

    elif decode=='simple':
        from qupy.ldpc.decoder import SimpleDecoder
        decoder = SimpleDecoder(code)

    elif decode=='schoning':
        from qupy.ldpc.decoder import SchoningDecoder
        decoder = SchoningDecoder(code)

    elif decode=='ml':
        decoder = MLDecoder(code)

    elif decode=='ensemble':
        C = argv.get("C", 40)
        K = argv.get("K", 1)
        decoder = ensemble.EnsembleDecoder(code, C, K)

    elif decode=='multiensemble':
        C = argv.get("C", 40)
        K = argv.get("K", 5)
        decoder = ensemble.MultiEnsembleDecoder(code, C, K)

    elif decode.startswith('lp'):
        lp_type = None
        if decode.endswith('int'):
            lp_type = int
        decoder = lp.LPDecoder(code, dense=decode.startswith("lpdense"), lp_type=lp_type)

    elif decode=='exact':
        from qupy.ldpc.mps import ExactDecoder
        decoder = ExactDecoder(code)

    elif decode=='oe':
        from qupy.ldpc.mps import OEDecoder
        decoder = OEDecoder(code)

    elif decode=='logop':
        from qupy.ldpc.mps import LogopDecoder
        decoder = LogopDecoder(code)

    elif decode=='mps':
        from qupy.ldpc.mps import MPSDecoder
        decoder = MPSDecoder(code)

    elif decode=='mpsensemble':
        from qupy.ldpc.mps import MPSEnsembleDecoder
        decoder = MPSEnsembleDecoder(code)

    elif decode=='lookup':
        decoder = LookupDecoder(code)

    return decoder


def whack(code, decoder, p, N, C0, error_rate, mC, verbose, argv):
    """keep decoding until we get below error_rate"""

    errors = []
    succeeds = [] # put error ops here when we decode them
    fails = [] # if the decoder is wrong

    for i in range(N):
        err_op = ra.binomial(1, p, (code.n,))
        err_op = err_op.astype(numpy.int32)
        errors.append(err_op)

    target_succeed = (1. - error_rate) * N

    i = 0
    C = C0

    while len(succeeds) < target_succeed and len(succeeds)+len(errors) >= target_succeed:

        err_op = errors[i]

        # We use Hz to look at X type errors (bitflip errors)

        write(str(err_op.sum()))

        s = dot2(code.Hz, err_op)
        write(":s%d:"%s.sum())

        op = decoder.decode(p, err_op, C=C, verbose=verbose, argv=argv)

        c = 'F'
        success = False
        if op is None:
            print(i, "TRY AGAIN")
            i = i+1

        else:
            op = (op+err_op)%2
            # Should be a codeword of Hz (kernel of Hz)
            assert dot2(code.Hz, op).sum() == 0
            write("%d:"%op.sum())

            success = code.is_stab(op)
            c = '.' if success else 'x'

            errors.pop(i)

            if success:
                print(i, "GOOD")
                succeeds.append(err_op)
            else:
                print(i, "BAD")
                fails.append(err_op)

        if i==len(errors):
            i = 0
            C = int(ceil(C*mC))
            print(("C = ", C))

        write(c+' ')

        print("errors = ", len(errors), "succeeds =", len(succeeds), "fails =", len(fails))

    if len(succeeds) >= target_succeed:
        print("SUCCESS")

    else:
        assert len(succeeds)+len(errors) < target_succeed
        print("FAIL")

    print()
    print(datestr)
    print(argv)
    print("error rate = %.8f" % (1.*(len(errors) + len(fails))/N))
    print("fail rate = %.8f" % (1.*len(fails) / N))


if __name__ == "__main__":

    _seed = argv.get('seed')
    if _seed is not None:
        seed(_seed)
        ra.seed(_seed)

    if argv.profile:
        import cProfile as profile
        profile.run("main()")

    else:
        main()


