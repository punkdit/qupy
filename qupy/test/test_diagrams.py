#!/usr/bin/env python

import sys
import math
import numpy

from qupy.dense import Qu, Gate, Vector
from qupy.dense import genidx, bits, is_close, on, off, scalar
from qupy.dense import commutator, anticommutator

from qupy.diagrams import cleanup, CharMatrix, parse
from qupy.test import test


def test_charmatrix():
    diagram = """
    --.--
      |
    --.--
      |
    --+--
    """

    assert cleanup(diagram) == '--.--\n  |\n--.--\n  |\n--+--\n'
    a = CharMatrix(diagram)
    assert a.rows, a.cols == (5, 5)
    assert str(a) == '--.--\n  |  \n--.--\n  |  \n--+--\n'




def test_parse_diagrams():

    #raise test.Skip

    I, X, Y, Z, H = Gate.I, Gate.X, Gate.Y, Gate.Z, Gate.H

    A = parse("""
    -
    """)
    assert A == I

    A = parse("""
    --
    """)
    assert A == I

    A = parse("""
    -
    -
    """)
    assert A == I*I

    A = parse("""
    --
    --
    """)
    assert A == I*I

    A = parse("""
    -H-
    --Z
    """)
    assert A == H*Z

    cnot = parse("""
    --.--
      |
    --+--
    """)
    assert cnot == Gate.CN

    cnot_1 = parse("""
    --+--
      |
    --.--
    """)

    swap = parse("""
    --x--
      |
    --x--
    """)
    assert swap == Gate.SWAP

    swap = parse("""
    --x--
      |
    -----
      |
    --x--
    """)
    assert swap|off*I*off == off*I*off
    assert swap|on*I*off == off*I*on
    assert swap|off*I*on == on*I*off
    assert swap|on*I*on == on*I*on

    encode_bell = parse("""
    -H--.--
        |
    ----+--
    """)

    toffoli = parse("""
    --.--
      |
    --.--
      |
    --+--
    """)
    assert toffoli|off*off*I == off*off*I
    assert toffoli|on*off*I == on*off*I
    assert toffoli|off*on*I == off*on*I
    assert toffoli|on*on*I == on*on*X

    assert toffoli == X.control(2, 1, 0)

    A = parse("""
    --.--
      |
    --Z--
      |
    --.--
    """)
    assert A|off*I*off == off*I*off
    assert A|on*I*off == on*I*off
    assert A|off*I*on == off*I*on
    assert A|on*I*on == on*Z*on

    assert A == Z.control(1, 0, 2)

    # Exercise 4.20 from C&N
    another_cnot = parse("""
    -H--+-H-
        |
    -H--.-H-
    """)
    assert another_cnot == Gate.CN

    # Controlled operation (Exercise 4.18 from C&N)
    cZ = parse("""
    --.--
      |
    --Z--
    """)
    assert cZ == Gate.Z.control()

    cZ = parse("""
    --Z--
      |
    --.--
    """)
    assert cZ == Gate.Z.control()

    # using an ancilla... meh, this ket looks silly (it's the
    # wrong way around)
#    """
#    |0>--H-+-H-
#           | 
#    -------X---
#    """

    A = parse("""
    --+--
      |
    -----
      |
    --.--
    """)
    assert A | (I*I*off) == I*I*off
    assert A | (I*I*on) == X*I*on

    A = parse("""
    --.--
      |
    --+--
      |
    --+--
    """)
    assert A | (off*I*I) == off*I*I
    assert A | (on*I*I) == on*X*X

    A = parse("""
    --+--
      |
    --.--
      |
    --+--
    """)
    assert A | (I*off*I) == I*off*I
    assert A | (I*on*I) == X*on*X

    A = parse("""
    --.--
      |
    --H--
      |
    --H--
    """)
    assert A | (off*I*I) == off*I*I
    assert A | (on*I*I) == on*H*H

    X12 = parse("""
    --.--
      |
    --X--
       
    -----
    """)

    X13 = parse("""
    --.--
      |
    -----
      |
    --X--
    """)

    X23 = parse("""
    -----
       
    --.--
      |
    --X--
    """)

    assert (X12|X13) == (X13|X12)
    assert (X13|X23) == (X23|X13)

    assert (X13|X23) | (X23|X13) == I*I*I


def test_parse_shor_encoder():

    I, X, Y, Z, H = Gate.I, Gate.X, Gate.Y, Gate.Z, Gate.H

    shor = """
    --.--.--H--.--.---
      |  |     |  |   
    -----------+------
      |  |        |   
    --------------+---
      |  |            
    --+-----H--.--.---
         |     |  |   
    -----------+------
         |        |   
    --------------+---
         |            
    -----+--H--.--.---
               |  |   
    -----------+------
                  |   
    --------------+---
    """

    SHOR = parse(shor)
    SHOR = SHOR | I*off*off*off*off*off*off*off*off

    # phase-flip encoder
    P = parse("""
    --H--.--.--
         |  |  
    -----+-----
            |  
    --------+--
    """)

    P = P|I*off*off # ancilla bits

    # bit-flip encoder
    B = parse("""
    --.--.--
      |  |  
    --+-----
         |  
    -----+--
    """)
    B = B|I*off*off # ancilla

    if 0:
        # Boo.. P has three outputs not one :-(
        S = parse("""
        --.--.--P--
          |  |  
        --+-----P--
             |  
        -----+--P--
        """, locals())

    S = P*P*P | B

    r2 = math.sqrt(2)
    x0 = (1./(2*r2)) * (bits('000') + bits('111')) * (bits('000') + bits('111')) * (bits('000') + bits('111'))
    x1 = (1./(2*r2)) * (bits('000') - bits('111')) * (bits('000') - bits('111')) * (bits('000') - bits('111'))

    assert S|off == x0
    assert S|on == x1

    assert S == SHOR

