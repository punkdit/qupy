#!/usr/bin/env python3

from qupy.abstract import *

def test_abstract():

    shape = (2, 2)
    valence = 'ud'

    A = AbstractQu(shape, valence)
    B = AbstractQu(shape, valence)

    C = A|B

    assert isinstance(C, LazyOrQu)
    assert C.shape == shape
    assert C.valence == valence

    I = IdentityQu(shape, valence)

    assert I|A is A

    assert A.dag().valence == 'du'

    B = A | A.dag()

    #shape = (2, 2, 2, 2)
    #valence = "udud"
    #A = PermutationQu(shape, valence, (0, 2, 1, 3))




