#!/usr/bin/env python3

import numpy

EPSILON = 1e-8

def swap_row(A, j, k):
    row = A[j, :].copy()
    A[j, :] = A[k, :]
    A[k, :] = row


def swap_col(A, j, k):
    col = A[:, j].copy()
    A[:, j] = A[:, k]
    A[:, k] = col


def row_reduce(A, truncate=True, inplace=False, check=False, 
    verbose=False, epsilon=EPSILON):
    """ Remove zero rows if truncate==True
    """

    A = numpy.array(A)

    assert len(A.shape)==2, A.shape
    m, n = A.shape
    if not inplace:
        A = A.copy()

    if m*n==0:
        if truncate and m:
            A = A[:0, :]
        return A

    if verbose:
        print("row_reduce")
        #print("%d rows, %d cols" % (m, n))

    i = 0
    j = 0
    while i < m and j < n:
        if verbose:
            print("i, j = %d, %d" % (i, j))
            print("A:")
            print(shortstrx(A))

        assert i<=j
        if i and check:
            assert (numpy.abs(A[i:,:j])>epsilon).sum() == 0

        # first find a nonzero entry in this col
        for i1 in range(i, m):
            if abs(A[i1, j])>epsilon:
                break
        else:
            j += 1 # move to the next col
            continue # <----------- continue ------------

        if i != i1:
            if verbose:
                print("swap", i, i1)
            swap_row(A, i, i1)

        assert abs(A[i, j]) > epsilon
        for i1 in range(i+1, m):
            if abs(A[i1, j])>epsilon:
                if verbose:
                    print("add row %s to %s" % (i, i1))
                r = -A[i1, j] / A[i, j]
                A[i1, :] += r*A[i, :]
                assert abs(A[i1, j]) < epsilon

        i += 1
        j += 1

    if truncate:
        m = A.shape[0]-1
        #print("sum:", m, A[m, :], A[m, :].sum())
        while m>=0 and (numpy.abs(A[m, :])>epsilon).sum()==0:
            m -= 1
        A = A[:m+1, :]

    if verbose:
        print()

    return A


def kernel(A, tol=EPSILON):
    assert len(A.shape)==2
    A = row_reduce(A) # does not change the kernel
    #print("kernel: A.shape=%s"%(A.shape,))
    u, s, vh = numpy.linalg.svd(A)
    #tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    #print("kernel: ns.shape=%s"%(ns.shape,))
    return ns



