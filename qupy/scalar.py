#!/usr/bin/env python3

import numpy

scalar = numpy.complex128

def use_reals():

    global scalar
    scalar = numpy.float64

EPSILON = 1e-8

MAX_TENSOR = 32

MAX_GATE_SIZE = 13 # conservative




