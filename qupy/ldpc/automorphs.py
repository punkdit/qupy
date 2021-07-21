#!/usr/bin/env python3

"""
Here are a few examples of codes based on {5,5} and {7,7} tessellations
of hyperbolic surfaces.  The files come as numpy matrices and include
X/Z parity check matrices and a basis of X/Z logicals).  The file names
are the schlaefli symbol, number of physical qubits and the minimum
distance.

55_32_5_Hx.npz    55_32_5_logZ.npz  55_360_8_logX.npz  77_78_5_Hz.npz
55_32_5_Hz.npz    55_360_8_Hx.npz   55_360_8_logZ.npz  77_78_5_logX.npz
55_32_5_logX.npz  55_360_8_Hz.npz   77_78_5_Hx.npz     77_78_5_logZ.npz
"""

import numpy
from scipy.sparse import csc_matrix, load_npz

from qupy.argv import argv
from qupy.ldpc.solve import int_scalar, shortstr, rank, linear_independent
from qupy.ldpc.cell import Complex


def write(*args):
    print(*args, end="", flush=True)

def load(name, verbose=False):
    print("load(%r)"%name)
    if verbose:
        blob = numpy.load(name)
        for a in blob.files:
            print(a)
            print(blob[a])
            print(blob[a].shape)

    H = load_npz(name).toarray().astype(int_scalar)
    H = H.transpose()
    return H

if argv.surface or argv.torus:

    # try the surface code...
    rows, cols = argv.get("rows", 3), argv.get("cols", 3)
    print(rows, cols)

    cx = Complex()
    if argv.surface:
        print("build_surface")
        cx.build_surface((0, 0), (rows, cols), open_top=True, open_bot=True)
    else:
        print("build_torus")
        cx.build_torus(rows, cols)
    code = cx.get_code()
    
    Hz, Hx = cx.get_parity_checks()
    print("Hz:")
    print(shortstr(Hz))
    print("Hx:")
    print(shortstr(Hx))

else:

    stem = argv.get("stem", "55_32_5")
    #stem = "55_360_8"
    #stem = "77_78_5"
    
    Hx = load(stem + "_Hx.npz")
    #Hx = linear_independent(Hx)
    print(Hx.shape)
    print(shortstr(Hx))
    #print(Hx.sum(0), Hx.sum(1))
    print(rank(Hx))
    
    Hz = load(stem + "_Hz.npz")
    Hz = linear_independent(Hz)
    print(Hz.shape)
    #print(shortstr(Hz))


from qupy.condmat.isomorph import Tanner, search

lhs = Tanner.build2(Hx, Hz)
if argv.duality:
    print("searching for duality")
    rhs = Tanner.build2(Hz, Hx)
else:
    rhs = Tanner.build2(Hx, Hz)

count = 0
for f in search(lhs, rhs):
    write('.')
    count += 1

print("\nautomorphism count:", count)



