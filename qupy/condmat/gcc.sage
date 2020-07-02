#!/usr/bin/env sage

F.<a,b,c,d> = FreeGroup(4)
rels = [  
          [ 24, [ a^2, b^2, c^2, (c*a)^2, (c*b)^3, (b*a)^3, (b*a*c)^4 ] ], 
          [ 96, [ a^2, b^2, c^2, d^2, (c*a)^2, (d*b)^2, (a*b)^3, (b*c)^3, (c*d)^3, (d*a)^3, b*d*a*c*b*d*c*a ] ], 
          [ 192, [ a^2, b^2, c^2, d^2, (a*c)^2, (b*d)^2, (b*c)^3, (b*a)^3, (c*d)^3, (d*a)^3, (b*a*c*d*a*c)^2 ] ], 
          [ 384, [ a^2, b^2, c^2, d^2, (a*c)^2, (b*d)^2, (a*b)^3, (b*c)^3, (c*d)^3, (d*a)^3, (b*c*d*a)^3 ] ], 
          [ 648, [ a^2, b^2, c^2, d^2, (a*c)^2, (b*d)^2, (b*c)^3, (b*a)^3, (c*d)^3, (d*a)^3, b*c*d*a*b*d*c*b*a*d*a*c*b*c*a*d*c*a ] ], 
          [ 768, [ a^2, b^2, c^2, d^2, (c*a)^2, (d*b)^2, (a*b)^3, (b*c)^3, (c*d)^3, (d*a)^3, (b*d*a*c)^2*(b*d*c*a)^2 ] ] 
       ]

import itertools as it


def get_color_code(R):
    """
    INPUT: Relations of the symmetry group of the lattice
    OUTPUT: GF(2) matrix with rows containing the indicator functions of the stabilizer generators
    """
    G = F/R
    G = G.as_permutation_group()
    gens = G.gens()
    
    colors = []
    for i in range(len(gens)):
        stab_gens = [x for j,x in enumerate(gens) if j != i]
        S = G.subgroup(stab_gens)
        colors.append(G.cosets(S, side='left'))
    enum = dict([(g,i) for i,g in enumerate(G.iteration())])
    colors = [[[enum[g] for g in cos] for cos in c] for c in colors]
    
    checks = [cos for c in colors for cos in c]
    nchecks = len(checks)
    n = G.order()
    
    H = matrix(GF(2),nchecks,n)
    for i,cos in enumerate(checks):
        for j in cos:
            H[i,j] = 1
    return H


def get_gauge_matrix(R):
    """
    INPUT: Relations of the symmetry group of the lattice
    OUTPUT: GF(2) matrix with rows containing the indicator functions of the gauge generators
    """
    G = F/R
    G = G.as_permutation_group()
    gens = G.gens()
    
    gauge_ops = []
    for i,j in it.combinations(range(len(gens)),2):
        stab_gens = [x for k,x in enumerate(gens) if k != i and k != j]
        S = G.subgroup(stab_gens)
        gauge_ops += G.cosets(S, side='left')
    enum = dict([(g,i) for i,g in enumerate(G.iteration())])
    gauge_ops = [[enum[g] for g in cos] for cos in gauge_ops]
    
    checks = gauge_ops
    nchecks = len(checks)
    n = G.order()
    
    H = matrix(GF(2),nchecks,n)
    for i,cos in enumerate(checks):
        for j in cos:
            H[i,j] = 1
    return H


def dump(H):
    H = str(H)
    H = H.replace(" ", "")
    H = H.replace("[", "")
    H = H.replace("]", "")
    H = H.replace("0", ".")
    print(H)

n, R = rels[3]

H = get_color_code(R)
#print("H =")
#dump(H)

stabweights = set([H.row(i).hamming_weight() for i in range(H.dimensions()[0])])
self_orthogonal = len((H*H.T).nonzero_positions())==0
#print(stabweights, self_orthogonal)

G = get_gauge_matrix(R)

set([r.hamming_weight() for r in G])

print("G =")
dump(G)


