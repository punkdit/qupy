#!/bin/sh

cython3 _algebra.pyx 

gcc -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -fPIC -I/usr/local/include/python3.8 \
    -c _algebra.c -o _algebra.o

gcc -shared _algebra.o -o _algebra.so


./test_algebra.py


