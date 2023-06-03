from time import time
import numpy as np
import numba
from numba import jit, cuda, float64, guvectorize, cfunc

## 1. Basic Loop Convolution
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

## 2. Loop + Numba convolution
@jit
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

@jit(nopython=False, forceobj=True)
def CPU_loop_numba_ConvolObject(A,F):
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

@jit(nopython=True)
def CPU_loop_numba_ConvolNJit(A,F):
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

@jit(float64[:,:](float64[:,:], float64[:,:]))
def CPU_loop_numba_ConvolEager(A,F):
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

@jit(float64[:,:](float64[:,:], float64[:,:]), nopython=True)
def CPU_loop_numba_ConvolEagerNJit(A,F):
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

@jit(nogil=True)
def CPU_loop_numba_ConvolNoGil(A,F):
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

@jit(cache=True)
def CPU_loop_numba_ConvolCache(A,F):
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

@jit(parallel=True)
def CPU_loop_numba_ConvolParallel(A,F):
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

# Mirar esto para guvectorize:

# https://github.com/anilsathyan7/ConvAcc/blob/master/conv_acc.ipynb

# NrA, NcA = A.shape
# NrF, NcF = F.shape

# assert NrA-NrF >= 1
# assert NcA-NcF >= 1

# C = np.zeros((NrA-NrF+1,NcA-NcF+1))

# @guvectorize([(float64[:,:], float64[:,:], float64[:,:])], '(n,m),(m,p)->(n,m)', nopython=True)
# def CPU_loop_numba_ConvolGuVectorize(A,F,C):
#     global Nra,Nrf,NcA,NcF
#     for i in range(NrA-NrF+1):
#         for j in range(NcA-NcF+1):
#             c = 0.0
#             for k in range(NrF):
#                 for t in range(NcF):
#                     c += A[i+k, j+t] * F[k,t]
#             C[i,j] = c

@cfunc("float64[:,:](float64[:,:], float64[:,:])")
def CPU_loop_numba_ConvolCFunc(A,F):
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

@jit(fastmath = True)
def CPU_loop_numba_ConvolFastMath(A,F):
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

## 3. Numpy Convolution
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

# 4. Numpy + Numba convolution
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

# 5. (GPU) Loop + Numba Convolution
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

# 7. (GPU) Shared memory convolution



########################################
### Main
########################################

# Common:
R = 30

N = 128
M = 112

# MatMult and MatConvol vars:
A = np.random.randn(N,M)  #.astype(np.float32)
#B = np.random.randn(M,N)  #.astype(np.float32)
C = np.zeros([A.shape[0],A.shape[1]])

K = 4
F = np.random.randn(K,K)  #.astype(np.float32)

#################
threadsperblock = (16, 16) # each block will contain 16×16 threads, typically 128 - 512 threads/bl
blockspergrid_x = int(np.ceil(C.shape[0] / threadsperblock[0]))
blockspergrid_y = int(np.ceil(C.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)
print(blockspergrid)
print(f"The kernel will be executed up to element {threadsperblock[0]*blockspergrid_x}")

TPB = 16

########################################
### Matrix multiplication

print("====================================================")

# CPU - numpy
tic = time()
for i in range(R):
    C = CPU_numpy_Convol(A,F)
print(" Convol - CPU - numpy:         {}".format(time() - tic))
print(C[0,0])

# CPU - numpy + numba
C = CPU_numpy_numba_Convol(A,F)  ### Compilation
tic = time()
for i in range(R):
    C = CPU_numpy_numba_Convol(A,F)
print(" Convol - CPU - numpy + numba: {}".format(time() - tic))
print(C[0,0])

# CPU - loop
tic = time()
for i in range(R):
    C = CPU_loop_Convol(A,F)
print(" Convol - CPU - loop:          {}".format(time() - tic))
print(C[0,0])

################ Numba ######################

# CPU - loop + numba (jit)
C = CPU_loop_numba_Convol(A,F)   ### Compilation
tic = time()
for i in range(R):
    C = CPU_loop_numba_Convol(A,F)
print(" Convol - CPU - loop + numba (jit):  {}".format(time() - tic))
print(C[0,0])

# CPU - loop + numba (Object Mode)
C = CPU_loop_numba_ConvolObject(A,F)   ### Compilation
tic = time()
for i in range(R):
    C = CPU_loop_numba_ConvolObject(A,F)
print(" Convol - CPU - loop + numba (Object):  {}".format(time() - tic))
print(C[0,0])

# CPU - loop + numba (njit)
C = CPU_loop_numba_ConvolNJit(A,F)   ### Compilation
tic = time()
for i in range(R):
    C = CPU_loop_numba_ConvolNJit(A,F)
print(" Convol - CPU - loop + numba (njit):  {}".format(time() - tic))
print(C[0,0])

# CPU - loop + numba (Eager)
C = CPU_loop_numba_ConvolEager(A,F)   ### Compilation
tic = time()
for i in range(R):
    C = CPU_loop_numba_ConvolEager(A,F)
print(" Convol - CPU - loop + numba (Eager):  {}".format(time() - tic))
print(C[0,0])

# CPU - loop + numba (Eager, NJit)
C = CPU_loop_numba_ConvolEagerNJit(A,F)   ### Compilation
tic = time()
for i in range(R):
    C = CPU_loop_numba_ConvolEagerNJit(A,F)
print(" Convol - CPU - loop + numba (Eager, NJit):  {}".format(time() - tic))
print(C[0,0])

# CPU - loop + numba (NoGil)
C = CPU_loop_numba_ConvolNoGil(A,F)   ### Compilation
tic = time()
for i in range(R):
    C = CPU_loop_numba_ConvolNoGil(A,F)
print(" Convol - CPU - loop + numba (NoGil):  {}".format(time() - tic))
print(C[0,0])

# CPU - loop + numba (Cache)
C = CPU_loop_numba_ConvolCache(A,F)   ### Compilation
tic = time()
for i in range(R):
    C = CPU_loop_numba_ConvolCache(A,F)
print(" Convol - CPU - loop + numba (Cache):  {}".format(time() - tic))
print(C[0,0])

# CPU - loop + numba (Parallel)
C = CPU_loop_numba_ConvolParallel(A,F)   ### Compilation
tic = time()
for i in range(R):
    C = CPU_loop_numba_ConvolParallel(A,F)
print(" Convol - CPU - loop + numba (Parallel):  {}".format(time() - tic))
print(C[0,0])

# # CPU - loop + numba (GuVectorize)
# CPU_loop_numba_ConvolGuVectorize(A,F,C)   ### Compilation
# tic = time()
# for i in range(R):
#     CPU_loop_numba_ConvolGuVectorize(A,F,C)
# print(" Convol - CPU - loop + numba (GuVectorize):  {}".format(time() - tic))
# print(C[0,0])

# CPU - loop + numba (CFunc)
C = CPU_loop_numba_ConvolCFunc(A,F)   ### Compilation
tic = time()
for i in range(R):
    C = CPU_loop_numba_ConvolCFunc(A,F)
print(" Convol - CPU - loop + numba (CFunc):  {}".format(time() - tic))
print(C[0,0])

# CPU - loop + numba (FastMath)
C = CPU_loop_numba_ConvolFastMath(A,F)   ### Compilation
tic = time()
for i in range(R):
    C = CPU_loop_numba_ConvolFastMath(A,F)
print(" Convol - CPU - loop + numba (FastMath):  {}".format(time() - tic))
print(C[0,0])


#############################################

# GPU - loop + numba
cA = cuda.to_device(A)
cF = cuda.to_device(F)
cC = cuda.to_device(np.zeros([A.shape[0],F.shape[1]]))
#
GPU_loop_numba_Convol[blockspergrid, threadsperblock](cA,cF,cC)   ### Compilation
tic = time()
for i in range(R):
    GPU_loop_numba_Convol[blockspergrid, threadsperblock](cA,cF,cC)
print(" Convol - GPU - loop + numba:  {}".format(time() - tic))
print(cC[0,0])

# # GPU - Shared Memory
# cA = cuda.to_device(A)
# cB = cuda.to_device(B)
# cC = cuda.to_device(np.zeros([A.shape[0],B.shape[1]]))
# GPU_loop_matMult_sharedMemory[blockspergrid, threadsperblock](cA,cB,cC)
# tic = time()
# for i in range(R):
#     GPU_loop_matMult_sharedMemory[blockspergrid, threadsperblock](cA,cB,cC)
# print(" MatMul - GPU - Shared Memory:  {}".format(time() - tic))
# print(cC[0,0])

# # GPU - Cupy
# tic = time()
# for i in range(R):
#     C = GPU_CuPy_matMult(A, B)
# print(" MatMul - GPU - Cupy1:  {}".format(time() - tic))
# print(C[0,0])
