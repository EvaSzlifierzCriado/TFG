from time import time
import numpy as np
import numba
from numba import jit, cuda, float64, guvectorize, cfunc
import matplotlib.pyplot as plt

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

N = 256
M = 256

# MatMult and MatConvol vars:
A = np.random.randn(N,M)  #.astype(np.float32)
#B = np.random.randn(M,N)  #.astype(np.float32)
C = np.zeros([A.shape[0],A.shape[1]])

K = 16
F = np.random.randn(K,K)  #.astype(np.float32)

#################
threadsperblock = (32, 32) # each block will contain 16×16 threads, typically 128 - 512 threads/bl
blockspergrid_x = int(np.ceil(C.shape[0] / threadsperblock[0]))
blockspergrid_y = int(np.ceil(C.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

print(f"The kernel will be executed up to element {threadsperblock[0]*blockspergrid_x}")
print("Threads per block:", threadsperblock)
print("Blocks per grid:", blockspergrid)
print(f"The kernel will be executed up to element {threadsperblock[0]*blockspergrid_x}")


########################################
######## Plot vars: ####################
########################################

x_comp = ["CPU - numpy + numba", "CPU - loop + numba (Jit)",
     #"CPU - loop + numba (Object)",
          "CPU - loop + numba (NJit)",
     "CPU - loop + numba (Eager)", "CPU - loop + numba (Eager, NJit)",
     "CPU - loop + numba (NoGil)", "CPU - loop + numba (Cache)",
     "CPU - loop + numba (Parallel)",
     #"CPU - loop + numba (CFunc)",
     "CPU - loop + numba (FastMath)",
     "GPU - loop + numba"]
x_exec = [#"CPU - Loop",
          "CPU - numpy", "CPU - numpy + numba", "CPU - loop + numba (Jit)",
     #"CPU - loop + numba (Object)",
          "CPU - loop + numba (NJit)",
     "CPU - loop + numba (Eager)", "CPU - loop + numba (Eager, NJit)",
     "CPU - loop + numba (NoGil)", "CPU - loop + numba (Cache)",
     "CPU - loop + numba (Parallel)",
     #"CPU - loop + numba (CFunc)",
     "CPU - loop + numba (FastMath)",
     "GPU - loop + numba"]
y_comp = []
y_exec = []

########################################
### Matrix multiplication

print("====================================================")

# # CPU - loop
# tic = time()
# for i in range(R):
#     C = CPU_loop_Convol(A,F)

# exec = time() - tic
# y_exec.append(exec)


# print(" Convol - CPU - loop:          {}", exec)
# print(C[0,0])

# CPU - numpy
tic = time()
for i in range(R):
    C = CPU_numpy_Convol(A,F)

exec = time() - tic
y_exec.append(exec)


print(" Convol - CPU - numpy:         {}", exec)
print(C[0,0])

# CPU - numpy + numba
tic = time()
C = CPU_numpy_numba_Convol(A,F)  ### Compilation
tic2 = time()
for i in range(R):
    C = CPU_numpy_numba_Convol(A,F)

exec = time() - tic2
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)


print(" Convol - CPU - numpy + numba (exec): {}", exec)
print(" Convol - CPU - numpy + numba: {}", comp)
print(C[0,0])

################ Numba ######################

# CPU - loop + numba (jit)
tic = time()
C = CPU_loop_numba_Convol(A,F)   ### Compilation
tic2 = time()
for i in range(R):
    C = CPU_loop_numba_Convol(A,F)

exec = time() - tic2
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)


print(" Convol - CPU - loop + numba (jit) (exec):  {}", exec)
print(" Convol - CPU - loop + numba (jit):  {}", comp)
print(C[0,0])

# # CPU - loop + numba (Object Mode)
# tic = time()
# C = CPU_loop_numba_ConvolObject(A,F)   ### Compilation
# tic2 = time()
# for i in range(R):
#     C = CPU_loop_numba_ConvolObject(A,F)

# exec = time() - tic2
# comp = time() - tic

# y_exec.append(exec)
# y_comp.append(comp)


# print(" Convol - CPU - loop + numba (Object) (exec):  {}", exec)
# print(" Convol - CPU - loop + numba (Object):  {}", comp)
# print(C[0,0])

# CPU - loop + numba (njit)
tic = time()
C = CPU_loop_numba_ConvolNJit(A,F)   ### Compilation
tic2 = time()
for i in range(R):
    C = CPU_loop_numba_ConvolNJit(A,F)

exec = time() - tic2
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)


print(" Convol - CPU - loop + numba (njit) (exec):  {}", exec)
print(" Convol - CPU - loop + numba (njit):  {}", comp)
print(C[0,0])

# CPU - loop + numba (Eager)
tic = time()
C = CPU_loop_numba_ConvolEager(A,F)   ### Compilation
tic2 = time()
for i in range(R):
    C = CPU_loop_numba_ConvolEager(A,F)

exec = time() - tic2
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)


print(" Convol - CPU - loop + numba (Eager) (exec):  {}", exec)
print(" Convol - CPU - loop + numba (Eager):  {}", comp)
print(C[0,0])

# CPU - loop + numba (Eager, NJit)
tic = time()
C = CPU_loop_numba_ConvolEagerNJit(A,F)   ### Compilation
tic2 = time()
for i in range(R):
    C = CPU_loop_numba_ConvolEagerNJit(A,F)

exec = time() - tic2
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)


print(" Convol - CPU - loop + numba (Eager, NJit) (exec):  {}", exec)
print(" Convol - CPU - loop + numba (Eager, NJit):  {}", comp)
print(C[0,0])

# CPU - loop + numba (NoGil)
tic = time()
C = CPU_loop_numba_ConvolNoGil(A,F)   ### Compilation
tic2 = time()
for i in range(R):
    C = CPU_loop_numba_ConvolNoGil(A,F)

exec = time() - tic2
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)


print(" Convol - CPU - loop + numba (NoGil) (exec):  {}", exec)
print(" Convol - CPU - loop + numba (NoGil):  {}", comp)
print(C[0,0])

# CPU - loop + numba (Cache)
tic = time()
C = CPU_loop_numba_ConvolCache(A,F)   ### Compilation
tic2 = time()
for i in range(R):
    C = CPU_loop_numba_ConvolCache(A,F)

exec = time() - tic2
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)


print(" Convol - CPU - loop + numba (Cache) (exec):  {}", exec)
print(" Convol - CPU - loop + numba (Cache):  {}", comp)
print(C[0,0])

# CPU - loop + numba (Parallel)
tic = time()
C = CPU_loop_numba_ConvolParallel(A,F)   ### Compilation
tic2 = time()
for i in range(R):
    C = CPU_loop_numba_ConvolParallel(A,F)

exec = time() - tic2
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)


print(" Convol - CPU - loop + numba (Parallel) (exec):  {}", exec)
print(" Convol - CPU - loop + numba (Parallel):  {}", comp)
print(C[0,0])

# # CPU - loop + numba (GuVectorize)
# CPU_loop_numba_ConvolGuVectorize(A,F,C)   ### Compilation
# tic = time()
# for i in range(R):
#     CPU_loop_numba_ConvolGuVectorize(A,F,C)
# print(" Convol - CPU - loop + numba (GuVectorize):  {}".format(time() - tic))
# print(C[0,0])

# # CPU - loop + numba (CFunc)
# tic = time()
# C = CPU_loop_numba_ConvolCFunc(A,F)   ### Compilation
# tic2 = time()
# for i in range(R):
#     C = CPU_loop_numba_ConvolCFunc(A,F)

# exec = time() - tic2
# comp = time() - tic

# y_exec.append(exec)
# y_comp.append(comp)


# print(" Convol - CPU - loop + numba (CFunc) (exec):  {}", exec)
# print(" Convol - CPU - loop + numba (CFunc):  {}", comp)
# print(C[0,0])

# CPU - loop + numba (FastMath)
tic = time()
C = CPU_loop_numba_ConvolFastMath(A,F)   ### Compilation
tic2 = time()
for i in range(R):
    C = CPU_loop_numba_ConvolFastMath(A,F)

exec = time() - tic2
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)


print(" Convol - CPU - loop + numba (FastMath) (exec):  {}", exec)
print(" Convol - CPU - loop + numba (FastMath):  {}", comp)
print(C[0,0])


#############################################

# GPU - loop + numba
cA = cuda.to_device(A)
cF = cuda.to_device(F)
cC = cuda.to_device(np.zeros([A.shape[0],F.shape[1]]))
#
tic = time()
GPU_loop_numba_Convol[blockspergrid, threadsperblock](cA,cF,cC)   ### Compilation
tic2 = time()
for i in range(R):
    GPU_loop_numba_Convol[blockspergrid, threadsperblock](cA,cF,cC)

exec = time() - tic2
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)


print(" Convol - GPU - loop + numba (exec):  {}", exec)
print(" Convol - GPU - loop + numba:  {}", comp)
print(cC[0,0])

# Plot printing:

# Convert the results to miliseconds:
for i in y_exec:
  i = i*1000

plt.figure(figsize=(10, 6))
plt.vlines(x_exec, 0, y_exec, linestyle="dashed", colors="green")
plt.scatter(x_exec, y_exec, c="green")
plt.xticks(x_exec, rotation=45, ha='right')
plt.title("Tiempos de ejecución - Convolución de matrices 256x256 16x16", pad=20)
plt.yscale('log')
plt.ylabel("Tiempo en ms", labelpad=20)
plt.xlabel("Implementación", labelpad=10)
plt.show()

# Execution and compilation time plot:

# Convert the results to miliseconds:
for i in y_comp:
  i = i*1000

plt.figure(figsize=(10, 6))
plt.vlines(x_comp, 0, y_comp, linestyle="dashed", colors="green")
plt.scatter(x_comp, y_comp, c="green")
plt.xticks(x_comp, rotation=45, ha='right')
plt.title("Tiempos de compilación y ejecución - Convolución de matrices 256x256 16x16", pad=20)
plt.yscale('log')
plt.ylabel("Tiempo en ms", labelpad=20)
plt.xlabel("Implementación", labelpad=10)
plt.show()

print(len(y_exec))
print(len(y_comp))
