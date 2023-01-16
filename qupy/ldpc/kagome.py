#!/usr/bin/env python

# process data from bruhat.comp 

import numpy

from qupy.ldpc import solve
from qupy.ldpc.solve import parse, shortstr, array2, zeros2, eq2, compose2, dot2
from qupy.ldpc.solve import intersect
from qupy.ldpc.chain import Chain, Morphism, pushout, equalizer
from qupy.argv import argv


def main(srcs, tgt, fs):

#    for i in [-1, 0, 1, 2]:
#        print(len(tgt.homology(i)))

    srcs = srcs[:3]
    fs = fs[:3]

    # check these are disjoint cover
    n = tgt[1].shape[0]
    vec = zeros2(1, n)
    for cmap in fs:
        f = cmap[1]
        vec += f.sum(1)
    assert numpy.alltrue(vec==1)

    dims = len(srcs)
    print("dims =", dims)

    #for src in srcs:
    #    print(src.homology(0))

    if 0:
        tgt = tgt.get_dual()
        srcs = [src.get_dual() for src in srcs]
        fs = [f.get_dual() for f in fs]
        
        f, g, h = fs
        
        f1, g1, code, _ = pushout(f, g)
        print(code) # Chain(0 <--- 0 <--- 9)
    
        return
    
    
    codes = [src.get_code() for src in srcs]
    Lxs = []
    
    if 1:
        for code in codes:
            print(code)
            #print(code.distance()) # (3, 6)
        
            LxHx = numpy.concatenate((code.Lx, code.Hx))
            Lxs.append(LxHx)
            #Lxs.append(code.Hx)
    
    else:
        for src in srcs:
            Lxs.append(src[0])
    
    LLs = []
    LLLs = []
    for i in range(dims):
      print(Lxs[i].shape)
      for j in range(i+1, dims):
        LL = intersect(Lxs[i], Lxs[j])
        print("\t", LL.shape)
        LLs.append(LL)
        for k in range(j+1, dims):
            LLL = intersect(LL, Lxs[k])
            print('\t\t', LLL.shape)
            LLLs.append(LLL)
    
    print(shortstr(LLL))
    


if __name__ == "__main__":

    if argv.kagome_2:
        from qupy.ldpc import kagome_2 as kagome
    elif argv.kagome_3:
        from qupy.ldpc import kagome_3 as kagome
    else:
        assert 0

    main(kagome.srcs, kagome.tgt, kagome.fs)

    print("OK\n")







