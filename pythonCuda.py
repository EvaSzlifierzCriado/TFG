
###
### numba
###   https://numba.readthedocs.io/en/stable/
###   (OLD) https://numba.readthedocs.io/en/stable/user/index.html
###
### numba + numpy
###   https://numba.readthedocs.io/en/stable/reference/numpysupported.html#
###
### numba + CUDA
###   https://numba.pydata.org/numba-doc/0.13/CUDAJit.html
###   https://developer.nvidia.com/how-to-cuda-python
###   https://nyu-cds.github.io/python-numba/05-cuda/
###

###
### anaconda environment
###
###   conda create -n numba
###   conda activate numba
###   conda install numba scipy cudatoolkit
###   conda update --all
###

from time import time
import numpy as np
import numba
from numba import jit, cuda   ### njit, vectorization


########################################
### Functions for Matrix Multiplication

def CPU_numpy_MatMul(A,B):
    return np.dot(A,B)


@jit(nopython=True)
def CPU_numpy_numba_MatMul(A,B):  ### Identical to CPU_numpy_MatMul but for the decorator
    return np.dot(A,B)


def CPU_loop_MatMul(A,B):
    NrA, NcA = A.shape
    NrB, NcB = B.shape

    assert NcA == NrB

    C = np.zeros((NrA,NcB))
    for i in range(NrA):
        for j in range(NcB):
            c = 0.0
            for k in range(NcA):
                c += A[i,k] * B[k,j]
            C[i,j] = c
    
    return C


@jit(nopython=False, forceobj=True)
def CPU_loop_numba_MatMul(A,B):
    NrA, NcA = A.shape
    NrB, NcB = B.shape

    assert NcA == NrB

    C = np.zeros((NrA,NcB))
    for i in range(NrA):
        for j in range(NcB):
            c = 0.0
            for k in range(NcA):
                c += A[i,k] * B[k,j]
            C[i,j] = c
    
    return C


@cuda.jit
def GPU_loop_numba_MatMul(A,B,C):
    NrA, NcA = A.shape
    NrB, NcB = B.shape

    assert NcA == NrB

    for i in range(NrA):
        for j in range(NcB):
            c = 0.0
            for k in range(NcA):
                c += A[i,k] * B[k,j]
            C[i,j] = c

###


########################################
### Functions for Convolution

def CPU_numpy_Convol(A,F):
    NrA, NcA = A.shape
    NrF, NcF = F.shape

    assert NrA-NrF >= 1
    assert NcA-NcF >= 1
    
    C = np.zeros((NrA-NrF+1,NcA-NcF+1))
    for i in range(NrA-NrF+1):
        for j in range(NcA-NcF+1):
            A1 = A[i:i+NrF, j:j+NcF]
            FM = np.multiply(A1,F)
            C[i,j] = np.sum(FM)

    return C


@jit(nopython=True)
def CPU_numpy_numba_Convol(A,F):  ### Identical to CPU_numpy_Convol but for the decorator
    NrA, NcA = A.shape
    NrF, NcF = F.shape

    assert NrA-NrF >= 1
    assert NcA-NcF >= 1

    C = np.zeros((NrA-NrF+1,NcA-NcF+1))
    for i in range(NrA-NrF+1):
        for j in range(NcA-NcF+1):
            A1 = A[i:i+NrF, j:j+NcF]
            FM = np.multiply(A1,F)
            C[i,j] = np.sum(FM)

    return C


def CPU_loop_Convol(A,F):
    NrA, NcA = A.shape
    NrF, NcF = F.shape

    assert NrA-NrF >= 1
    assert NcA-NcF >= 1

    C = np.zeros((NrA-NrF+1,NcA-NcF+1))
    for i in range(NrA-NrF+1):
        for j in range(NcA-NcF+1):
            c = 0.0
            for k in range(NrF):
                for t in range(NcF):
                    c += A[i+k, j+t] * F[k,t]
            C[i,j] = c
    
    return C


@jit(nopython=False, forceobj=True)
def CPU_loop_numba_Convol(A,F):
    NrA, NcA = A.shape
    NrF, NcF = F.shape

    assert NrA-NrF >= 1
    assert NcA-NcF >= 1

    C = np.zeros((NrA-NrF+1,NcA-NcF+1))
    for i in range(NrA-NrF+1):
        for j in range(NcA-NcF+1):
            c = 0.0
            for k in range(NrF):
                for t in range(NcF):
                    c += A[i+k, j+t] * F[k,t]
            C[i,j] = c
    
    return C


@cuda.jit
def GPU_loop_numba_Convol(A,F,C):
    NrA, NcA = A.shape
    NrF, NcF = F.shape

    assert NrA-NrF >= 1
    assert NcA-NcF >= 1

    for i in range(NrA-NrF+1):
        for j in range(NcA-NcF+1):
            c = 0.0
            for k in range(NrF):
                for t in range(NcF):
                    c += A[i+k, j+t] * F[k,t]
            C[i,j] = c

###


########################################
### Main
########################################


R = 30

N = 100
M = 50
A = np.random.randn(N,M)  #.astype(np.float32)
B = np.random.randn(M,N)  #.astype(np.float32)

K = 4
F = np.random.randn(K,K)  #.astype(np.float32)

griddim  = 8, 8
blockdim = 16, 16



########################################
### Matrix multiplication

print("====================================================")

### CPU - numpy
tic = time()
for i in range(R):
    C = CPU_numpy_MatMul(A,B)
print(" MatMul - CPU - numpy:         {}".format(time() - tic))
print(C[0,0])

### CPU - numpy + numba
C = CPU_numpy_numba_MatMul(A,B)  ### Compilation
tic = time()
for i in range(R):
    C = CPU_numpy_numba_MatMul(A,B)
print(" MatMul - CPU - numpy + numba: {}".format(time() - tic))
print(C[0,0])

### CPU - loop
tic = time()
for i in range(R):
    C = CPU_loop_MatMul(A,B)
print(" MatMul - CPU - loop:          {}".format(time() - tic))
print(C[0,0])

### CPU - loop + numba
C = CPU_loop_numba_MatMul(A,B)   ### Compilation
tic = time()
for i in range(R):
    C = CPU_loop_numba_MatMul(A,B)
print(" MatMul - CPU - loop + numba:  {}".format(time() - tic))
print(C[0,0])

### GPU - loop + numba
cA = cuda.to_device(A)
cB = cuda.to_device(B)
cC = cuda.to_device(np.zeros([A.shape[0],B.shape[1]]))
#
GPU_loop_numba_MatMul[griddim, blockdim](cA,cB,cC)   ### Compilation
tic = time()
for i in range(R):
    GPU_loop_numba_MatMul[griddim, blockdim](cA,cB,cC)
print(" MatMul - GPU - loop + numba:  {}".format(time() - tic))
print(cC[0,0])



########################################
### Convolution

print("====================================================")

### CPU - numpy
tic = time()
for i in range(R):
    C = CPU_numpy_Convol(A,F)
print(" Convol - CPU - numpy:         {}".format(time() - tic))
print(C[0,0])

### CPU - numpy + numba
C = CPU_numpy_numba_Convol(A,F)  ### Compilation
tic = time()
for i in range(R):
    C = CPU_numpy_numba_Convol(A,F)
print(" Convol - CPU - numpy + numba: {}".format(time() - tic))
print(C[0,0])

### CPU - loop
tic = time()
for i in range(R):
    C = CPU_loop_Convol(A,F)
print(" Convol - CPU - loop:          {}".format(time() - tic))
print(C[0,0])

### CPU - loop + numba
C = CPU_loop_numba_Convol(A,F)   ### Compilation
tic = time()
for i in range(R):
    C = CPU_loop_numba_Convol(A,F)
print(" Convol - CPU - loop + numba:  {}".format(time() - tic))
print(C[0,0])

### GPU - loop + numba
cA = cuda.to_device(A)
cF = cuda.to_device(F)
cC = cuda.to_device(np.zeros((A.shape[0]-F.shape[0]+1,A.shape[1]-F.shape[1]+1)))
#
GPU_loop_numba_Convol[griddim, blockdim](cA,cF,cC)   ### Compilation
tic = time()
for i in range(R):
    GPU_loop_numba_Convol[griddim, blockdim](cA,cF,cC)
print(" Convol - GPU - loop + numba:  {}".format(time() - tic))
print(cC[0,0])

########################################

print("====================================================")
