from time import time
import numpy as np
import cupy as cp
import numba
from numba import jit, cuda 

## Matrix Multiplication

## 1. (CPU) Basic loop multiplication -> 0.0016710758209228516
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

## 2. (CPU) Loop + Numba multiplication -> 1.109595537185669
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

## 3. (CPU) NumPy multiplication -> 0.005754232406616211
def CPU_numpy_MatMul(A,B):
    return np.dot(A,B)

## 4. (CPU) NumPy + Numba multiplication -> 0.6577882766723633
@jit(nopython=True)
def CPU_numpy_numba_MatMul(A,B):  ### Identical to CPU_numpy_MatMul but for the decorator
    return np.dot(A,B)

## 5. (GPU) loop + Numba multiplication -> 0.8816361427307129
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

## 6. (GPU) loop multiplication 
@cuda.jit
def GPU_loop_matMult(A, B, C):
  i, j = cuda.grid(2)
  if i < C.shape[0] and j < C.shape[1]:
    tmp = 0. 
    for k in range(A.shape[1]):
      tmp += A[i, k] * B[k, j]
    C[i, j] = tmp

# 7. (GPU) loop + shared memory multiplication (reduces de number of times that each number is send to the GPU)
# (Not working rn)
TPB = 2

@cuda.jit
def GPU_loop_matMult_sharedMemory(A, B, C):
  sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
  sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
  x, y = cuda.grid()

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


## 8. (GPU) CuPy multiplication
def GPU_CuPy_matMult(A, B):
    a_gpu = cp.asarray(A)
    b_gpu = cp.asarray(B)

    c_gpu = cp.dot(a_gpu, b_gpu)

    return cp.asnumpy(c_gpu)

def main():
  A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
  B = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)
  C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)

  threads_per_block = (16, 16)
  blocks_per_grid_x = int(np.ceil(A.shape[0] / threads_per_block[0]))
  blocks_per_grid_y = int(np.ceil(B.shape[1] / threads_per_block[1]))
  blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

  GPU_loop_matMult[blocks_per_grid, threads_per_block](A, B, C)
  print(C)

if __name__ == '__main__':
  main()




























# # Example matrices to try the code.
# m1 = [[1, 2], [3, 4]]
# m2 = [[5, 6], [7, 8]]
# m3 = [[9, 10], [11, 12]]

# # Given a list of matrixes M, returns the multiplication of all
# # the matrices of the list using a loop. 
# def matrixMult(M):
#     result = M[0]
#     for i in range(1, len(M)):
#         matrix = M[i]
#         if len(result[0]) != len(matrix): # C1 must be equal to F2
#             raise ValueError('Matrices cannot be multiplicated')
#         newResult = []
#         for j in range(len(result)):
#             row = []
#             for k in range(len(matrix[0])):
#                 element = 0
#                 for l in range(len(matrix)):
#                     element += result[j][l] * matrix[l][k]
#                 row.append(element)
#             newResult.append(row)
#         result = newResult
#     return result

# # Matrix convolution

# # Función de partición exacta

# # Aproximación de de la función de partición con AIS


# def main():
#     print(matrixMult([m1, m2, m3]))
    
# main()