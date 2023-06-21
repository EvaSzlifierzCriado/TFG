from time import time
import numpy as np
# import cupy as cp
import numba
from numba import jit, cuda, float64, guvectorize, cfunc
from numpy import float32
import matplotlib.pyplot as plt

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

## 6. (GPU) loop + shared memory multiplication (reduces de number of times that each number is send to the GPU)
@cuda.jit
def GPU_loop_matMult_sharedMemoryV3(A, B, C):
  TPB = M

  # Create a shared memory matrix for each block
  sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
  sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

  # Thread position in the grid
  x, y = cuda.grid(2)

  # Thread position in the block
  tx = cuda.threadIdx.x
  ty = cuda.threadIdx.y

  ## Number of bloocks per grid
  #bpg = cuda.gridDim.x # = 3

  # Checking thread inside the grid
  if x >= C.shape[0] and y >= C.shape[1]:
    return

  sA[tx, ty] = A[x, ty]  # A[x, ty + cuda.blockIdx.y * TPB]
  sB[tx, ty] = B[tx, y]  # B[tx + cuda.blockIdx.x * TPB, y]

  # 1st. Syncronize threads of the block after loading the data into the shared memory
  cuda.syncthreads()

  tmp = 0.
  for j in range(TPB):
      tmp += sA[tx, j] * sB[j, ty]

  # 2nd. Syncronize threads after calculation
  #cuda.syncthreads()

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

N = 10
M = 5

# MatMult and MatConvol vars:
A = np.random.randn(N,M)  #.astype(np.float32)
B = np.random.randn(M,N)  #.astype(np.float32)
C = np.zeros([A.shape[0],B.shape[1]])

# Testing that the matrix multiplication can be computed
NrA, NcA = A.shape
NrB, NcB = B.shape

assert NcA == NrB

#################
threadsperblock = (M, M) # each block will contain 16×16 threads, typically 128 - 512 threads/bl
blockspergrid_x = int(np.ceil(C.shape[0] / threadsperblock[0]))
blockspergrid_y = int(np.ceil(C.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

print(f"The kernel will be executed up to element {threadsperblock[0]*blockspergrid_x}")


########################################
######## Plot vars: ####################
########################################

x_comp = ["CPU - numpy + numba", "CPU - loop + numba (Jit)",
     "CPU - loop + numba (Object)", "CPU - loop + numba (NJit)", 
     "CPU - loop + numba (Eager)", "CPU - loop + numba (Eager, NJit)",
     "CPU - loop + numba (NoGil)", "CPU - loop + numba (Cache)",
     "CPU - loop + numba (Parallel)", "CPU - loop + numba (GuVectorize)",
     "CPU - loop + numba (CFunc)", "CPU - loop + numba (FastMath)",
     "GPU - loop + numba", "GPU - Shared Memory"]
x_exec = ["CPU - Loop", "CPU - numpy", "CPU - numpy + numba", "CPU - loop + numba (Jit)",
     "CPU - loop + numba (Object)", "CPU - loop + numba (NJit)", 
     "CPU - loop + numba (Eager)", "CPU - loop + numba (Eager, NJit)",
     "CPU - loop + numba (NoGil)", "CPU - loop + numba (Cache)",
     "CPU - loop + numba (Parallel)", "CPU - loop + numba (GuVectorize)",
     "CPU - loop + numba (CFunc)", "CPU - loop + numba (FastMath)",
     "GPU - loop + numba", "GPU - Shared Memory"]
y_comp = []
y_exec = []


########################################
### Matrix multiplication

print("====================================================")

# CPU - loop
tic = time()
for i in range(R):
    C = CPU_loop_MatMul(A,B)

exec = time() - tic
y_exec.append(exec)

print(" MatMul - CPU - loop:          ", exec)
print(C[0,0])

# CPU - numpy
tic = time()
for i in range(R):
    C = CPU_numpy_MatMul(A,B)

exec = time() - tic
y_exec.append(exec)

print(" MatMul - CPU - numpy:         ", exec)
print(C[0,0])

# CPU - numpy + numba
tic = time()
C = CPU_numpy_numba_MatMul(A,B)  ### Compilation
tic2 = time()
for i in range(R):
    C = CPU_numpy_numba_MatMul(A,B)

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)

print(" MatMul - CPU - numpy + numba (exec): {}", exec)
print(" MatMul - CPU - numpy + numba: {}", comp)
print(C[0,0])

################ Numba ######################

# CPU - loop + numba (jit)
tic = time()
C = CPU_loop_numba_MatMul(A,B)   ### Compilation
tic2 = time()
for i in range(R):
    C = CPU_loop_numba_MatMul(A,B)

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)


print(" MatMul - CPU - loop + numba (jit) (exec):  {}", exec)
print(" MatMul - CPU - loop + numba (jit):  {}", comp)
print(C[0,0])

# CPU - loop + numba (Object Mode)
tic = time()
C = CPU_loop_numba_MatMulObject(A,B)   ### Compilation
tic2 = time()
for i in range(R):
    C = CPU_loop_numba_MatMulObject(A,B)

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)


print(" MatMul - CPU - loop + numba (Object) (exec):  {}", exec)
print(" MatMul - CPU - loop + numba (Object):  {}", comp)
print(C[0,0])

# CPU - loop + numba (njit)
tic = time()
C = CPU_loop_numba_MatMulNJit(A,B)   ### Compilation
tic2 = time()
for i in range(R):
    C = CPU_loop_numba_MatMulNJit(A,B)

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)


print(" MatMul - CPU - loop + numba (njit) (exec):  {}", exec)
print(" MatMul - CPU - loop + numba (njit):  {}", comp)
print(C[0,0])

# CPU - loop + numba (Eager)
tic = time()
C = CPU_loop_numba_MatMulEager(A,B)   ### Compilation
tic2 = time()
for i in range(R):
    C = CPU_loop_numba_MatMulEager(A,B)

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)


print(" MatMul - CPU - loop + numba (Eager) (exec):  {}", exec)
print(" MatMul - CPU - loop + numba (Eager):  {}", comp)
print(C[0,0])

# CPU - loop + numba (Eager, NJit)
tic = time()
C = CPU_loop_numba_MatMulEagerNJit(A,B)   ### Compilation
tic2 = time()
for i in range(R):
    C = CPU_loop_numba_MatMulEagerNJit(A,B)

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)


print(" MatMul - CPU - loop + numba (Eager, NJit) (exec):  {}", exec)
print(" MatMul - CPU - loop + numba (Eager, NJit):  {}", comp)
print(C[0,0])

# CPU - loop + numba (NoGil)
tic = time()
C = CPU_loop_numba_MatMulNoGil(A,B)   ### Compilation
tic2 = time()
for i in range(R):
    C = CPU_loop_numba_MatMulNoGil(A,B)

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)


print(" MatMul - CPU - loop + numba (NoGil) (exec):  {}", exec)
print(" MatMul - CPU - loop + numba (NoGil):  {}", comp)
print(C[0,0])

# CPU - loop + numba (Cache)
tic = time()
C = CPU_loop_numba_MatMulCache(A,B)   ### Compilation
tic2 = time()
for i in range(R):
    C = CPU_loop_numba_MatMulCache(A,B)

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)


print(" MatMul - CPU - loop + numba (Cache) (exec):  {}", exec)
print(" MatMul - CPU - loop + numba (Cache):  {}", comp)
print(C[0,0])

# CPU - loop + numba (Parallel)
tic = time()
C = CPU_loop_numba_MatMulParallel(A,B)   ### Compilation
tic2 = time()
for i in range(R):
    C = CPU_loop_numba_MatMulParallel(A,B)

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)


print(" MatMul - CPU - loop + numba (Parallel) (exec):  {}", exec)
print(" MatMul - CPU - loop + numba (Parallel):  {}", comp)
print(C[0,0])

# CPU - loop + numba (GuVectorize)
tic = time()
CPU_loop_numba_MatMulGuVectorize(A,B,C)   ### Compilation
tic2 = time()
for i in range(R):
    CPU_loop_numba_MatMulGuVectorize(A,B,C)

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)


print(" MatMul - CPU - loop + numba (GuVectorize) (exec):  {}", exec)
print(" MatMul - CPU - loop + numba (GuVectorize):  {}", comp)
print(C[0,0])

# CPU - loop + numba (CFunc)
tic = time()
C = CPU_loop_numba_MatMulCFunc(A,B)   ### Compilation
tic2 = time()
for i in range(R):
    C = CPU_loop_numba_MatMulCFunc(A,B)

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)


print(" MatMul - CPU - loop + numba (CFunc) (exec):  {}", exec)
print(" MatMul - CPU - loop + numba (CFunc):  {}", comp)
print(C[0,0])

# CPU - loop + numba (FastMath)
tic = time()
C = CPU_loop_numba_MatMulFastMath(A,B)   ### Compilation
tic2 = time()
for i in range(R):
    C = CPU_loop_numba_MatMulFastMath(A,B)

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)


print(" MatMul - CPU - loop + numba (FastMath) (exec):  {}", exec)
print(" MatMul - CPU - loop + numba (FastMath):  {}", comp)
print(C[0,0])


#############################################

# GPU - loop + numba
cA = cuda.to_device(A)
cB = cuda.to_device(B)
cC = cuda.to_device(np.zeros([A.shape[0],B.shape[1]]))
tic = time()
GPU_loop_numba_matMult[blockspergrid, threadsperblock](cA,cB,cC)   ### Compilation
tic2 = time()
for i in range(R):
    GPU_loop_numba_matMult[blockspergrid, threadsperblock](cA,cB,cC)

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)


print(" MatMul - GPU - loop + numba (exec):  {}", exec)
print(" MatMul - GPU - loop + numba:  {}", comp)
print(cC[0,0])

# GPU - Shared Memory
cA = cuda.to_device(A)
cB = cuda.to_device(B)
cC = cuda.to_device(np.zeros([A.shape[0],B.shape[1]]))
tic = time()
GPU_loop_matMult_sharedMemoryV3[blockspergrid, threadsperblock](cA,cB,cC)
tic2 = time()
for i in range(R):
    GPU_loop_matMult_sharedMemoryV3[blockspergrid, threadsperblock](cA,cB,cC)

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)

print(" MatMul - GPU - Shared Memory v3 (exec):  {}", exec)
print(" MatMul - GPU - Shared Memory v3:  {}", comp)
print(cC[0,0])

# Plot printing:

# Execution time plot:

# Convert the results to miliseconds:
for i in y_exec:
  i = i*1000

plt.figure(figsize=(10, 6))
plt.vlines(x_exec, 0, y_exec, linestyle="dashed")
plt.scatter(x_exec, y_exec)
plt.xticks(x_exec, rotation=45, ha='right')
plt.title("Tiempos de ejecución - Multiplicación de matrices", pad=20)
plt.yscale('log')
plt.ylabel("Tiempo en ms", labelpad=20)
plt.xlabel("Implementación", labelpad=10)
plt.show()

# Execution and compilation time plot:

# Convert the results to miliseconds:
for i in y_comp:
  i = i*1000

plt.figure(figsize=(10, 6))
plt.vlines(x_comp, 0, y_comp, linestyle="dashed")
plt.scatter(x_comp, y_comp)
plt.xticks(x_comp, rotation=45, ha='right')
plt.title("Tiempos de compilación y ejecución - Multiplicación de matrices", pad=20)
plt.yscale('log')
plt.ylabel("Tiempo en ms", labelpad=20)
plt.xlabel("Implementación", labelpad=10)
plt.show()

print(len(y_exec))
print(len(y_comp))


