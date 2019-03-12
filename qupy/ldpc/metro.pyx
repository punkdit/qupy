

cdef extern from "math.h":
    double fabs(double)
    double log(double)
    double pow(double, double)


cdef extern from "stdlib.h":
    void *malloc(int)
    void *calloc(int, int) 
    void *realloc(void *, int) 
    void free(void *)
    long int random()
    int rand()
    void srand(unsigned int)
    int RAND_MAX
    void srandom(unsigned int seed)

cdef extern from "string.h":
    void *memset(void *s, int c, size_t n)
    void *memmove(void *, void *, size_t)
    void *memcpy(void *, void *, size_t)

import random as py_random
srandom(py_random.randint(0, RAND_MAX-1))


cdef int nrand(int n):
    cdef int j
    j = <int> (n * (rand() / (RAND_MAX + 1.0)))
    return j


cdef double rand_r():
    cdef double p
    p = <double>rand() / <double>RAND_MAX
    return p




cimport numpy as cnp
cnp.import_array()

import numpy


#def bufaddr(cnp.ndarray buffer):
#    data = <long>buffer.data
#    return data


cdef int sum(int *data, int n):
    cdef int i, count
    count = 0
    for i from 0<=i<n:
        if data[i]:
            count += 1
    return count


cdef void move(int *A, int *B, int n):
    cdef int i
    for i from 0<=i<n:
        A[i] = B[i]


cdef void iadd2(int *A, int *B, int n):
    cdef int i
    for i from 0<=i<n:
        A[i] = (A[i]+B[i])%2


def metropolis(double p, cnp.ndarray T, int N, cnp.ndarray Hx):

    cdef int m, n, count, n0, n1, idx, n_min
    cdef double r
    cdef int *T0_data, *T1_data, *Hx_data, *tmp
    cdef cnp.ndarray T1

    #m, n = Hx.shape

    m = Hx.shape[0]
    n = Hx.shape[1]
    assert Hx.dtype == numpy.int64
    Hx_data = <int *>Hx.data
    #print Hx.strides[0], Hx.strides[1]

    T0_data = <int *>T.data
    T1 = T.copy()
    T1_data = <int *>T1.data

    # Calculate an exponential moving average...
    cdef double alpha, avg
    alpha = min(0.1, 10./N)
    n0 = sum(T0_data, n)
    n_min = n0
    avg = n0

    count = 0
    while count<N: 

        idx = nrand(m)
        #stab = Hx[idx]
        iadd2(T1_data, Hx_data + idx*n, n)
        #T1 = (stab + T0)%2
        n1 = sum(T1_data, n)

        #r = (p/(1-p))**(n1-n0)
        r = pow(p/(1-p), n1-n0)
        if r >= 1 or rand_r() <= r:
            tmp = T0_data
            T0_data = T1_data
            T1_data = tmp
            n0 = n1

            if n0<n_min:
                n_min = n0

        count += 1
        #print avg, n0
        avg = alpha*n0 + (1.-alpha)*avg

    move(<int *>T.data, T0_data, n)

    # These all seem very similar:

    return n_min
    #return n0
    #return avg







