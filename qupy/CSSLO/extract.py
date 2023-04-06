#!/usr/bin/env python

#from add_parent_dir import *
from common import *
from NSpace import *
from clifford_LO import *
import itertools as iter

from bruhat.solve import shortstr


########################################################
## Triorthogonal spaces from Classification of Small Triorthogonal Codes
########################################################

## Level of Clifford Hierarchy
t = 3
## Precision of Codes
N = 1 << t
## number of logical qubits = |LX|
k = 3

triData = triCodeData()

print("#!/usr/bin/env python")
print("from bruhat.solve import parse")
print("codes = []")

for triRow in triData:

    res = getTriCode(triRow,k)
    assert res
    SX, LX = res
    k,n = np.shape(LX)
    # print(f't-Orthogonality: {tOrthogonal(np.vstack([SX,LX]))}')
    Eq, SX,LX,SZ,LZ = CSSCode(SX,LX)
    print('Hx = parse("""')
    print(shortstr(SX))
    print('""")')
    print('Hz = parse("""')
    print(shortstr(SZ))
    print('""")')
    print('Lx = parse("""')
    print(shortstr(LX))
    print('""")')
    print('Lz = parse("""')
    print(shortstr(LZ))
    print('""")')
    # print(np.sum(LX,axis=-1))
    zList,qList, V, K_M = comm_method(Eq, SX, LX, SZ, t, compact=True, debug=False)
    print('comment = """')
    print(f'# [[{n},{k}]] Triorthogonal code: ',TriCodeParams(triRow))
    print('# Transversal Diagonal Logical Operators')
    for z,q in zip(zList,qList):
        print("#", CP2Str(2*q,V,N),"=>",z2Str(z,N))
    print('"""')
    print("codes.append((Hx, Hz, Lx, Lz, comment))")

