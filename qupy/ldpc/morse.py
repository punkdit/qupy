#!/usr/bin/env python3

from qupy.ldpc.solve import parse, dot2
from qupy.ldpc.css import CSSCode


def main():

    n = 5
    Tx = parse("""
    1....
    1.1..
    1.1.1
    """)

    Tz = parse("""
    .1...
    ...1.
    """)


    code = CSSCode(Hx=Tx, Hz=Tz)

    #print(code)
    #print(code.longstr())

    S = parse("""
    1.1..1
    11.1..
    .11.1.
    ...111
    """)

    I = parse("""
    1...
    .1..
    ..1.
    ...1
    """)

    A = parse("""
    1...
    11..
    ..1.
    ...1
    """)

    B = parse("""
    1...
    .1..
    .11.
    ...1
    """)

    C = parse("""
    1...
    .1..
    ..1.
    ..11
    """)

    #print(I)
    #print(dot2(A))
    #print(dot2(B, A))
    L = dot2(C, B, A)
    #print(dot2(L, S)) # row reduced S

    L = L[:3, :3]

    S = S[:3, :]
    print("S =")
    print(S)

    print("LS =")
    print(dot2(L, S))

    J = parse("""
    1..
    11.
    ...
    ...
    111
    ...
    """)

    U = parse("""
    1..
    .11
    ..1
    """)

    print(dot2(U, L, S))



if __name__ == "__main__":

    main()


