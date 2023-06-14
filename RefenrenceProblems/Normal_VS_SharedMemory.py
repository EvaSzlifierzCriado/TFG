from time import time
import numpy as np
# import cupy as cp
import numba
from numba import jit, cuda, float64, guvectorize, cfunc
from numpy import float32
import matplotlib.pyplot as plt

# Comparation with plot (matplotlib)

# 1. (CPU) NumPy multiplication
def CPU_numpy_MatMul(A,B):
    return np.dot(A,B)

# 2. (GPU) loop multiplication 
@cuda.jit
def GPU_loop_numba_matMult(A, B, C):
  # Thread position in the grid
  i, j = cuda.grid(2) 

  # Check thread inside C
  if i < C.shape[0] and j < C.shape[1]: 
    tmp = 0. 
    for k in range(A.shape[1]):
      tmp += A[i, k] * B[k, j]
    C[i, j] = tmp 

## 3. V2 (GPU) loop + shared memory multiplication (reduces de number of times that each number is send to the GPU)
@cuda.jit
def GPU_loop_matMult_sharedMemoryV2(A, B, C):
  TPB = M

  # Create a shared memory matrix for each block
  sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
  sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

  # Thread position in the grid
  x, y = cuda.grid(2)

  # Thread position in the block
  tx = cuda.threadIdx.x
  ty = cuda.threadIdx.y

  # Number of bloocks per grid
  bpg = cuda.gridDim.x # = 3

  # Checking thread inside the grid
  if x >= C.shape[0] and y >= C.shape[1]:
    return

  tmp = 0.
  for i in range(bpg):
    if x < (i * TPB + TPB-1) and x > (i * (TPB-1)):
      if y < (i * TPB + TPB-1) and y > (i * (TPB-1)):
        sA[tx, ty] = A[x, ty + i * TPB] 
        sB[tx, ty] = B[tx + i * TPB, y] 

        # 1st. Syncronize threads after loading the data into the shared memory
        cuda.syncthreads()

        for j in range(TPB):
          tmp += sA[tx, j] * sB[j, ty]
        
        # 2nd. Syncronize threads after calculation
        cuda.syncthreads()

        C[x,y] = tmp

## 4. V1 (GPU) loop + shared memory multiplication (reduces de number of times that each number is send to the GPU)
@cuda.jit
def GPU_loop_matMult_sharedMemory(A, B, C):
  TPB = M

  # Create a shared memory matrix for each block
  sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
  sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

  # Thread position in the grid
  x, y = cuda.grid(2)

  # Thread position in the block
  tx = cuda.threadIdx.x
  ty = cuda.threadIdx.y

  # Number of bloocks per grid
  bpg = cuda.gridDim.x # = 3

  # Checking thread inside the grid
  if x >= C.shape[0] and y >= C.shape[1]:
    return

  tmp = 0.
  for i in range(bpg):
    sA[tx, ty] = A[x, ty + i * TPB] 
    sB[tx, ty] = B[tx + i * TPB, y] 

    # 1st. Syncronize threads after loading the data into the shared memory
    cuda.syncthreads()

    for j in range(TPB):
      tmp += sA[tx, j] * sB[j, ty]
    
    # 2nd. Syncronize threads after calculation
    cuda.syncthreads()

    C[x,y] = tmp

########################################
### Main
########################################

# Common:
R = 30

N = 10 # 128
M = 5 # 112

# MatMult and MatConvol vars:
A = np.random.randn(N,M).astype(np.float32)
B = np.random.randn(M,N).astype(np.float32)
C = np.zeros([A.shape[0],B.shape[1]]).astype(np.float32)

# Testing that the matrix multiplication can be computed
NrA, NcA = A.shape
NrB, NcB = B.shape

assert NcA == NrB

#################
threadsperblock = (M, M) # each block will contain 16×16 threads, typically 128 - 512 threads/bl
blockspergrid_x = int(np.ceil(C.shape[0] / threadsperblock[0]))
blockspergrid_y = int(np.ceil(C.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)
print("Threads per block:", threadsperblock)
print("Blocks per grid:", blockspergrid)
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

previous = C[0,0]

# GPU - loop + numba
cA = cuda.to_device(A)
cB = cuda.to_device(B)
cC = cuda.to_device(C)

GPU_loop_numba_matMult[blockspergrid, threadsperblock](cA,cB,cC) 
tic = time()
for i in range(R):
    GPU_loop_numba_matMult[blockspergrid, threadsperblock](cA,cB,cC)
print(" MatMul - GPU - loop + numba:  {}".format(time() - tic))
res = cC.copy_to_host()
print(res[0,0])

V1_count = 0 # (V1) Number of times that the result function differs from the actual result
V2_count = 0 # (V2) Number of times that the result function differs from the actual result

for aux in range(500):
  # GPU - Shared Memory V2
  cA = cuda.to_device(A)
  cB = cuda.to_device(B)
  cC = cuda.to_device(C)

  first = True

  GPU_loop_matMult_sharedMemoryV2[blockspergrid, threadsperblock](cA,cB,cC)
  tic = time()
  for i in range(R):
      GPU_loop_matMult_sharedMemoryV2[blockspergrid, threadsperblock](cA,cB,cC)
      if first == False:
        if previous != cC[0,0]:
          V2_count += 1
      first = False
  #print(" MatMul - GPU - Shared Memory V2:  {}".format(time() - tic))
  res = cC.copy_to_host()
  #print(res[0,0])

  # GPU - Shared Memory V1
  cA = cuda.to_device(A)
  cB = cuda.to_device(B)
  cC = cuda.to_device(C)

  first = True

  GPU_loop_matMult_sharedMemory[blockspergrid, threadsperblock](cA,cB,cC)
  tic = time()
  for i in range(R):
      GPU_loop_matMult_sharedMemory[blockspergrid, threadsperblock](cA,cB,cC)
      if first == False:
        if previous != cC[0,0]:
          V1_count += 1
      first = False
  #print(" MatMul - GPU - Shared Memory V1:  {}".format(time() - tic))
  res = cC.copy_to_host()
  #print(res[0,0])

print("Number of times that V1 fails:", V1_count)
print("Number of times that V2 fails:", V2_count)

x = ["V1 (Numba)", "V2"]
y = [V1_count, V2_count]
plt.bar(x,y)
plt.title("Comparation on the number of failures on V1 and V2")
plt.xlabel("Versions of the shared memory function")
plt.ylabel("Number of failures")
plt.show()
