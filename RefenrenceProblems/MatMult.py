from time import time
import numpy as np
# import cupy as cp
import numba
from numba import jit, cuda, float64, guvectorize, cfunc
from numpy import float32

# Decorators:
#     + @jit in no python (njit)
#     + Eager compilation
#     + @jit object mode 
#     + nogil = true
#     + cache = true
#     + parallel 
#     - vectorize
#     + @guvectorize
#     - @stencil
#     - @jitclass
#     + @cfunc
#     - @overload
#     - fastmath = True
#     - cffi
#     - ctypes
#     - @generated_jit
#     - jit_module

## Matrix Multiplication

# 1. (CPU) Basic loop multiplication -> 0.0016710758209228516
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

# # 2. (CPU) Loop + Numba multiplication -> 1.109595537185669
# @jit(nopython=False, forceobj=True)
# def CPU_loop_numba_MatMul(A,B):
#     NrA, NcA = A.shape
#     NrB, NcB = B.shape

#     assert NcA == NrB

#     C = np.zeros((NrA,NcB))
#     for i in range(NrA):
#         for j in range(NcB):
#             c = 0.0
#             for k in range(NcA):
#                 c += A[i,k] * B[k,j]
#             C[i,j] = c
    
#     return C


# @jit(nopython=True)
# def CPU_loop_numba_MatMulNJit(A,B):
#     NrA, NcA = A.shape
#     NrB, NcB = B.shape

#     assert NcA == NrB

#     C = np.zeros((NrA,NcB))
#     for i in range(NrA):
#         for j in range(NcB):
#             c = 0.0
#             for k in range(NcA):
#                 c += A[i,k] * B[k,j]
#             C[i,j] = c
    
#     return C


# @jit(float64[:,:](float64[:,:], float64[:,:]))
# def CPU_loop_numba_MatMulEager(A,B):
#     NrA, NcA = A.shape
#     NrB, NcB = B.shape

#     assert NcA == NrB

#     C = np.zeros((NrA,NcB))
#     for i in range(NrA):
#         for j in range(NcB):
#             c = 0.0
#             for k in range(NcA):
#                 c += A[i,k] * B[k,j]
#             C[i,j] = c
    
#     return C


# @jit(float64[:,:](float64[:,:], float64[:,:]), nopython=True)
# def CPU_loop_numba_MatMulEagerNJit(A,B):
#     NrA, NcA = A.shape
#     NrB, NcB = B.shape

#     assert NcA == NrB

#     C = np.zeros((NrA,NcB))
#     for i in range(NrA):
#         for j in range(NcB):
#             c = 0.0
#             for k in range(NcA):
#                 c += A[i,k] * B[k,j]
#             C[i,j] = c
    
#     return C


# @jit(nogil=True)
# def CPU_loop_numba_MatMulNoGil(A,B):
#     NrA, NcA = A.shape
#     NrB, NcB = B.shape

#     assert NcA == NrB

#     C = np.zeros((NrA,NcB))
#     for i in range(NrA):
#         for j in range(NcB):
#             c = 0.0
#             for k in range(NcA):
#                 c += A[i,k] * B[k,j]
#             C[i,j] = c
    
#     return C


# @jit(cache=True)
# def CPU_loop_numba_MatMulCache(A,B):
#     NrA, NcA = A.shape
#     NrB, NcB = B.shape

#     assert NcA == NrB

#     C = np.zeros((NrA,NcB))
#     for i in range(NrA):
#         for j in range(NcB):
#             c = 0.0
#             for k in range(NcA):
#                 c += A[i,k] * B[k,j]
#             C[i,j] = c
    
#     return C


# @jit(parallel=True)
# def CPU_loop_numba_MatMulParallel(A,B):
#     NrA, NcA = A.shape
#     NrB, NcB = B.shape

#     assert NcA == NrB

#     C = np.zeros((NrA,NcB))
#     for i in range(NrA):
#         for j in range(NcB):
#             c = 0.0
#             for k in range(NcA):
#                 c += A[i,k] * B[k,j]
#             C[i,j] = c
    
#     return C


################3 NO VA ################
# @guvectorize([(float64[:,:], float64[:,:], float64[:,:])], '(n,m),(n,m)->(n,m)')
# def CPU_loop_numba_MatMulGUVectorize(A,B, C):
#     NrA, NcA = A.shape
#     NrB, NcB = B.shape

#     assert NcA == NrB

#     C = np.zeros((NrA,NcB))
#     for i in range(NrA):
#         for j in range(NcB):
#             c = 0.0
#             for k in range(NcA):
#                 c += A[i,k] * B[k,j]
#             C[i,j] = c
    
# @cfunc("float64[:,:](float64[:,:], float64[:,:])")
# def CPU_loop_numba_MatMulCFunc(A,B):
#     NrA, NcA = A.shape
#     NrB, NcB = B.shape

#     assert NcA == NrB

#     C = np.zeros((NrA,NcB))
#     for i in range(NrA):
#         for j in range(NcB):
#             c = 0.0
#             for k in range(NcA):
#                 c += A[i,k] * B[k,j]
#             C[i,j] = c
    
#     return C


# 3. (CPU) NumPy multiplication -> 0.005754232406616211
def CPU_numpy_MatMul(A,B):
    return np.dot(A,B)

# 4. (CPU) NumPy + Numba multiplication -> 0.6577882766723633
@jit(nopython=True)
def CPU_numpy_numba_MatMul(A,B):  ### Identical to CPU_numpy_MatMul but for the decorator
    return np.dot(A,B)

# 5. (GPU) loop multiplication 
@cuda.jit
def GPU_loop_numba_matMult(A, B, C):
  i, j = cuda.grid(2)
  if i < C.shape[0] and j < C.shape[1]:
    tmp = 0. 
    for k in range(A.shape[1]):
      tmp += A[i, k] * B[k, j]
    C[i, j] = tmp

# ## 6. (GPU) loop + shared memory multiplication (reduces de number of times that each number is send to the GPU)
# # Matrix has to be  squared!!
# TPB = 2

# @cuda.jit
# def GPU_loop_matMult_sharedMemory(A, B, C):
#   sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
#   sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
#   x, y = cuda.grid(2)

#   tx = cuda.threadIdx.x
#   ty = cuda.threadIdx.y
#   bpg = cuda.gridDim.x 

#   if x >= C.shape[0] and y >= C.shape[1]:
#     return
#   tmp = 0.
#   for i in range(bpg):
#     sA[tx, ty] = A[x, ty + i * TPB]
#     sB[tx, ty] = B[tx + i * TPB, y]
#     cuda.syncthreads()
#     for j in range(TPB):
#       tmp += sA[tx, j] * sB[j, ty]
#       cuda.syncthreads()
#     C[x,y] = tmp

# ## 7. (GPU) CuPy multiplication
# def GPU_CuPy_matMult(A, B):
#     a_gpu = cp.asarray(A)
#     b_gpu = cp.asarray(B)

#     c_gpu = cp.dot(a_gpu, b_gpu)

#     return cp.asnumpy(c_gpu)

########################################
### Main
########################################

# Common:
R = 30

N = 128
M = 112

# MatMult and MatConvol vars:
A = np.random.randn(N,M)  #.astype(np.float32)
B = np.random.randn(M,N)  #.astype(np.float32)
C = np.zeros([A.shape[0],B.shape[1]])

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
    C = CPU_numpy_MatMul(A,B)
print(" MatMul - CPU - numpy:         {}".format(time() - tic))
print(C[0,0])

# CPU - numpy + numba
C = CPU_numpy_numba_MatMul(A,B)  ### Compilation
tic = time()
for i in range(R):
    C = CPU_numpy_numba_MatMul(A,B)
print(" MatMul - CPU - numpy + numba: {}".format(time() - tic))
print(C[0,0])

# CPU - loop
tic = time()
for i in range(R):
    C = CPU_loop_MatMul(A,B)
print(" MatMul - CPU - loop:          {}".format(time() - tic))
print(C[0,0])

# CPU - loop + numba
C = CPU_loop_numba_MatMul(A,B)   ### Compilation
tic = time()
for i in range(R):
    C = CPU_loop_numba_MatMul(A,B)
print(" MatMul - CPU - loop + numba:  {}".format(time() - tic))
print(C[0,0])

# GPU - loop + numba
cA = cuda.to_device(A)
cB = cuda.to_device(B)
cC = cuda.to_device(np.zeros([A.shape[0],B.shape[1]]))
#
GPU_loop_numba_matMult[blockspergrid, threadsperblock](cA,cB,cC)   ### Compilation
tic = time()
for i in range(R):
    GPU_loop_numba_matMult[blockspergrid, threadsperblock](cA,cB,cC)
print(" MatMul - GPU - loop + numba:  {}".format(time() - tic))
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
