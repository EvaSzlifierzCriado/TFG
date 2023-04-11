from time import time
import numpy as np
import numba
from numba import jit, cuda   ### njit, vectorization

# 1. Basic Loop Convolution
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

# 2. Loop + Numba convolution
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

# 3. Numpy Convolution
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
@jit(nopython=True)
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

# 6. (GPU) loop convolution
TPB = 16

@cuda.jit
def GPU_loop_Convol(A,F,C):
    NrA, NcA = A.shape
    NrF, NcF = F.shape

    assert NrA-NrF >= 1
    assert NcA-NcF >= 1
    
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        c = 0.0
        for k in range(NrF):
            for t in range(NcF):
                c += A[i+k, j+t] * F[k,t]
        C[i,j] = c

def CUDA_loop_Convol(A, F):
    NrA, NcA = A.shape
    NrF, NcF = F.shape

    assert NrA-NrF >= 1
    assert NcA-NcF >= 1

    C = np.zeros((NrA-NrF+1, NcA-NcF+1), dtype=np.float32)

    # allocate memory on the device
    d_A = cuda.to_device(A)
    d_F = cuda.to_device(F)
    d_C = cuda.to_device(C)

    # set up the kernel launch parameters
    griddim = (int(np.ceil((NrA - NrF + 1) / TPB)), int(np.ceil((NcA - NcF + 1) / TPB)))
    blockdim = (TPB, TPB)

    # launch the kernel on the device
    GPU_loop_Convol[griddim, blockdim](d_A, d_F, d_C)

    # copy the result back from the device and return it
    C = d_C.copy_to_host()


# 7. (GPU) Shared memory convolution


