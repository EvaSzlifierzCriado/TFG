from time import time
import numpy as np
# import cupy as cp
import numba
from numba import jit, cuda, float64, guvectorize, cfunc
from numpy import float32

# Decorators:
#     + jit:
#           La recomendada. Con esta opcion Numba decide 
#           Cuando y como optimizar
#     + @jit in no python (njit)
#     + @jit object mode 
#     + nogil = true
#     + cache = true
#     - vectorize
#     + @guvectorize
#     - @stencil:
#           No es util, se usa cuando queremos 
#           updatear cada elemento de un array con algun
#           pattern fijado llamado stencil kernel.
#     - @jitclass:
#           Se usa para clases.
#     + @cfunc
#     - @overload:
#           Es util cuando tenemos numpy + numba y numpy no
#           soporta el tipo. Por ejemplo, con integer arrays
#           usariamos overload + numpy ya que numpy por si 
#           solo no lo soporta.

# Extra options:
#     + parallel 
#     + fastmath = True
#     + Eager compilation
#           Especificarle de que tipos son los 
#           parametros para tener mas control.
#     - @generated_jit:
#           Es utiluando hay diferentes implementaciones 
#           dependiendodel tipo de datos que se pasan a 
#           la funcion.
#     - jit_module:
#           Es util cuando tenemos varias funciones que
#           usan jit (si lo usa una, y esta llama a las otras,
#           todas deberian usarlo). Es lo mismo que poner
#           cada decorador manualmente, pero quita trabajo
#           si hay muchas funciones.

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

# 2. (CPU) Loop + Numba multiplication 
@jit
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


@jit(nopython=False, forceobj=True)
def CPU_loop_numba_MatMulObject(A,B):
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


@jit(nopython=True)
def CPU_loop_numba_MatMulNJit(A,B):
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


@jit(float64[:,:](float64[:,:], float64[:,:]))
def CPU_loop_numba_MatMulEager(A,B):
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


@jit(float64[:,:](float64[:,:], float64[:,:]), nopython=True)
def CPU_loop_numba_MatMulEagerNJit(A,B):
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


@jit(nogil=True)
def CPU_loop_numba_MatMulNoGil(A,B):
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


@jit(cache=True)
def CPU_loop_numba_MatMulCache(A,B):
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


@jit(parallel=True)
def CPU_loop_numba_MatMulParallel(A,B):
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

@guvectorize([(float64[:,:], float64[:,:], float64[:,:])], '(n,m),(m,p)->(n,p)', nopython=True)
def CPU_loop_numba_MatMulGuVectorize(A, B, C):
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            s = 0
            for k in range(A.shape[1]):
                s += A[i,k] * B[k,j]
            C[i,j] = s
    
@cfunc("float64[:,:](float64[:,:], float64[:,:])")
def CPU_loop_numba_MatMulCFunc(A,B):
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

@jit(fastmath = True)
def CPU_loop_numba_MatMulFastMath(A,B):
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

@guvectorize(['void(float32[:,:], float32[:,:], float32[:,:])'],
             '(m,n),(n,p)->(m,p)', target='cuda')
def GPU_loop_numba_matMultGuVectorize(A, B, C):
  i, j = cuda.grid(2)
  if i < C.shape[0] and j < C.shape[1]:
    tmp = 0. 
    for k in range(A.shape[1]):
      tmp += A[i, k] * B[k, j]
    C[i, j] = tmp


## 6. (GPU) loop + shared memory multiplication (reduces de number of times that each number is send to the GPU)
# Matrix has to be  squared!!
TPB = 16

@cuda.jit
def GPU_loop_matMult_sharedMemory(A, B, C):
  sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
  sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
  x, y = cuda.grid(2)

  tx = cuda.threadIdx.x
  ty = cuda.threadIdx.y
  bpg = cuda.gridDim.x 

  if x >= C.shape[0] and y >= C.shape[1]:
    return
  tmp = 0.
  for i in range(bpg):
    sA[tx, ty] = A[x, ty + i * TPB]
    sB[tx, ty] = B[tx + i * TPB, y]
    cuda.syncthreads()
    for j in range(TPB):
      tmp += sA[tx, j] * sB[j, ty]
      cuda.syncthreads()
    C[x,y] = tmp

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
M = 128

# MatMult and MatConvol vars:
A = np.random.randn(N,M)  #.astype(np.float32)
B = np.random.randn(M,N)  #.astype(np.float32)
C = np.zeros([A.shape[0],B.shape[1]])

# Testing that the matrix multiplication can be computed
NrA, NcA = A.shape
NrB, NcB = B.shape

assert NcA == NrB

#################
threadsperblock = (16, 16) # each block will contain 16×16 threads, typically 128 - 512 threads/bl
blockspergrid_x = int(np.ceil(C.shape[0] / threadsperblock[0]))
blockspergrid_y = int(np.ceil(C.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)
print(blockspergrid)
print(f"The kernel will be executed up to element {threadsperblock[0]*blockspergrid_x}")

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

################ Numba ######################

# CPU - loop + numba (jit)
C = CPU_loop_numba_MatMul(A,B)   ### Compilation
tic = time()
for i in range(R):
    C = CPU_loop_numba_MatMul(A,B)
print(" MatMul - CPU - loop + numba (jit):  {}".format(time() - tic))
print(C[0,0])

# CPU - loop + numba (Object Mode)
C = CPU_loop_numba_MatMulObject(A,B)   ### Compilation
tic = time()
for i in range(R):
    C = CPU_loop_numba_MatMulObject(A,B)
print(" MatMul - CPU - loop + numba (Object):  {}".format(time() - tic))
print(C[0,0])

# CPU - loop + numba (njit)
C = CPU_loop_numba_MatMulNJit(A,B)   ### Compilation
tic = time()
for i in range(R):
    C = CPU_loop_numba_MatMulNJit(A,B)
print(" MatMul - CPU - loop + numba (njit):  {}".format(time() - tic))
print(C[0,0])

# CPU - loop + numba (Eager)
C = CPU_loop_numba_MatMulEager(A,B)   ### Compilation
tic = time()
for i in range(R):
    C = CPU_loop_numba_MatMulEager(A,B)
print(" MatMul - CPU - loop + numba (Eager):  {}".format(time() - tic))
print(C[0,0])

# CPU - loop + numba (NoGil)
C = CPU_loop_numba_MatMulNoGil(A,B)   ### Compilation
tic = time()
for i in range(R):
    C = CPU_loop_numba_MatMulNoGil(A,B)
print(" MatMul - CPU - loop + numba (NoGil):  {}".format(time() - tic))
print(C[0,0])

# CPU - loop + numba (Cache)
C = CPU_loop_numba_MatMulCache(A,B)   ### Compilation
tic = time()
for i in range(R):
    C = CPU_loop_numba_MatMulCache(A,B)
print(" MatMul - CPU - loop + numba (Cache):  {}".format(time() - tic))
print(C[0,0])

# CPU - loop + numba (Parallel)
C = CPU_loop_numba_MatMulParallel(A,B)   ### Compilation
tic = time()
for i in range(R):
    C = CPU_loop_numba_MatMulParallel(A,B)
print(" MatMul - CPU - loop + numba (Parallel):  {}".format(time() - tic))
print(C[0,0])

# CPU - loop + numba (GuVectorize)
CPU_loop_numba_MatMulGuVectorize(A,B,C)   ### Compilation
tic = time()
for i in range(R):
    CPU_loop_numba_MatMulGuVectorize(A,B,C)
print(" MatMul - CPU - loop + numba (GuVectorize):  {}".format(time() - tic))
print(C[0,0])

# CPU - loop + numba (Cache)
C = CPU_loop_numba_MatMulCFunc(A,B)   ### Compilation
tic = time()
for i in range(R):
    C = CPU_loop_numba_MatMulCFunc(A,B)
print(" MatMul - CPU - loop + numba (CFunc):  {}".format(time() - tic))
print(C[0,0])

# CPU - loop + numba (FastMath)
C = CPU_loop_numba_MatMulFastMath(A,B)   ### Compilation
tic = time()
for i in range(R):
    C = CPU_loop_numba_MatMulFastMath(A,B)
print(" MatMul - CPU - loop + numba (FastMath):  {}".format(time() - tic))
print(C[0,0])


#############################################

# GPU - loop + numba
cA = cuda.to_device(A)
cB = cuda.to_device(B)
cC = cuda.to_device(np.zeros([A.shape[0],B.shape[1]]))

GPU_loop_numba_matMult[blockspergrid, threadsperblock](cA,cB,cC)   ### Compilation
tic = time()
for i in range(R):
    GPU_loop_numba_matMult[blockspergrid, threadsperblock](cA,cB,cC)
print(" MatMul - GPU - loop + numba:  {}".format(time() - tic))
print(cC[0,0])

# cC = cuda.to_device(np.zeros([A.shape[0],B.shape[1]]))
# GPU_loop_numba_matMultGuVectorize.max_blocksize = 32  # limits to 32 threads per block

# GPU_loop_numba_matMultGuVectorize[blockspergrid, threadsperblock](cA,cB,cC)   ### Compilation
# tic = time()
# for i in range(R):
#     GPU_loop_numba_matMultGuVectorize[blockspergrid, threadsperblock](cA,cB,cC)
# print(" MatMul - GPU - loop + numba (GuVectorize):  {}".format(time() - tic))
# print(cC[0,0])

# GPU - Shared Memory
cA = cuda.to_device(A)
cB = cuda.to_device(B)
cC = cuda.to_device(np.zeros([A.shape[0],B.shape[1]]))
GPU_loop_matMult_sharedMemory[blockspergrid, threadsperblock](cA,cB,cC)
tic = time()
for i in range(R):
    GPU_loop_matMult_sharedMemory[blockspergrid, threadsperblock](cA,cB,cC)
print(" MatMul - GPU - Shared Memory:  {}".format(time() - tic))
print(cC[0,0])

# # GPU - Cupy
# tic = time()
# for i in range(R):
#     C = GPU_CuPy_matMult(A, B)
# print(" MatMul - GPU - Cupy1:  {}".format(time() - tic))
# print(C[0,0])


#################### TEST RESULTS ####################


# (8, 8)
# The kernel will be executed up to element 128
# ====================================================
#  MatMul - CPU - numpy:         0.019099950790405273
# -3.545629036135979
#  MatMul - CPU - numpy + numba: 0.02588510513305664
# -3.545629036135979
#  MatMul - CPU - loop:          31.282602787017822
# -3.5456290361359795
#  MatMul - CPU - loop + numba (jit):  0.09103178977966309
# -3.5456290361359795
#  MatMul - CPU - loop + numba (Object):  23.346181869506836
# -3.5456290361359795
#  MatMul - CPU - loop + numba (njit):  0.08579444885253906
# -3.5456290361359795
#  MatMul - CPU - loop + numba (Eager):  0.07909131050109863
# -3.5456290361359795
#  MatMul - CPU - loop + numba (NoGil):  0.07879400253295898
# -3.5456290361359795
#  MatMul - CPU - loop + numba (Cache):  0.08299708366394043
# -3.5456290361359795
#  MatMul - CPU - loop + numba (Parallel):  0.10923480987548828
# -3.5456290361359795
#  MatMul - CPU - loop + numba (GuVectorize):  0.09578061103820801
# -3.5456290361359795
#  MatMul - CPU - loop + numba (CFunc):  24.679866313934326
# -3.5456290361359795
#  MatMul - CPU - loop + numba (FastMath):  0.08434128761291504
# -3.545629036135979
#  MatMul - GPU - loop + numba:  0.0022399425506591797
# -3.545629036135979
#  MatMul - GPU - Shared Memory:  0.001954317092895508
# -3.5456292890849