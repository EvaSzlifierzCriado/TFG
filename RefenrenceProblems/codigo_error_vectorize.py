from time import time
import numpy as np
import cupy as cp
import numba
from numba import jit, cuda 
from numpy import float64
from numba import guvectorize, float64

# Common:
N = 128
M = 112

# MatMult vars:
A = np.random.randn(N,M).astype(np.float64)
B = np.random.randn(M,N).astype(np.float64)
C = np.zeros([A.shape[0],B.shape[1]], dtype=np.float64)

K = 4
F = np.random.randn(K,K).astype(np.float64)

print(np.dtype(A[0][0]))
print(np.dtype(B[0][0]))
print(np.dtype(C[0][0]))
print(np.dtype(F[0][0]))

@guvectorize([(float64[:,:], float64[:,:], float64[:,:])], '(n,m),(m,p)->(n,p)', nopython=True)
def dot_product(x, y, res):
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            s = 0
            for k in range(x.shape[1]):
                s += x[i,k] * y[k,j]
            res[i,j] = s


NrA, NcA = A.shape
NrB, NcB = B.shape

assert NcA == NrB

@jit(nopython=False, forceobj=True)
def CPU_loop_numba_MatMul(A,B):
    for i in range(NrA):
        for j in range(NcB):
            c = 0.0
            for k in range(NcA):
                c += A[i,k] * B[k,j]
            C[i,j] = c
    return C

tic = time()
for i in range(R):
    dot_product(A, B, C)
print(" MatMul - CPU - numba:         {}".format(time() - tic))
print(C[0,0])

C = np.zeros([A.shape[0],B.shape[1]])

tic = time()
for i in range(R):
    C = CPU_loop_numba_MatMul(A,B)
print(" MatMul - guvectorize:         {}".format(time() - tic))
print(C[0,0])
