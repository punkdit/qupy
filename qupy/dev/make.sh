#!/bin/sh

cython3 _algebra.pyx 

gcc -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/usr/include/python3.6 -I/usr/include/python3.5 \
    -c _algebra.c -o _algebra.o

gcc -shared _algebra.o -o _algebra.so


./test_algebra.py


