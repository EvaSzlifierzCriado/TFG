from time import time
import numpy as np
import numba
from numba import jit, cuda   ### njit, vectorization

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



# # 6. (GPU) loop convolution
# TPB = 16

# @cuda.jit
# def GPU_loop_Convol(A,F,C):
#     NrA, NcA = A.shape
#     NrF, NcF = F.shape

#     assert NrA-NrF >= 1
#     assert NcA-NcF >= 1
    
#     i, j = cuda.grid(2)
#     if i < C.shape[0] and j < C.shape[1]:
#         c = 0.0
#         for k in range(NrF):
#             for t in range(NcF):
#                 c += A[i+k, j+t] * F[k,t]
#         C[i,j] = c

# def CUDA_loop_Convol(A, F):
#     NrA, NcA = A.shape
#     NrF, NcF = F.shape

#     assert NrA-NrF >= 1
#     assert NcA-NcF >= 1

#     C = np.zeros((NrA-NrF+1, NcA-NcF+1), dtype=np.float32)

#     # allocate memory on the device
#     d_A = cuda.to_device(A)
#     d_F = cuda.to_device(F)
#     d_C = cuda.to_device(C)

#     # set up the kernel launch parameters
#     griddim = (int(np.ceil((NrA - NrF + 1) / TPB)), int(np.ceil((NcA - NcF + 1) / TPB)))
#     blockdim = (TPB, TPB)

#     # launch the kernel on the device
#     GPU_loop_Convol[griddim, blockdim](d_A, d_F, d_C)

#     # copy the result back from the device and return it
#     C = d_C.copy_to_host()


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

# GPU - Cupy
tic = time()
for i in range(R):
    C = GPU_CuPy_matMult(A, B)
print(" MatMul - GPU - Cupy1:  {}".format(time() - tic))
print(C[0,0])
