import numpy as np
from math import e
import numba.cuda as cuda
from time import time
import numba
from numba import jit, cuda, float64, guvectorize, cfunc
import matplotlib.pyplot as plt

# Scalar product of two arrays
def CPU_loop_ArrayMul(A,B):

    NcA = len(A)
    NrB = len(B)

    assert NcA == NrB

    res = 0
    for i in range(NcA):
        res += A[i]*B[i]
    return(res)

# Multiplies array A per matrix B
def CPU_loop_ArrayMatMul(A,B):

    NcA = len(A)
    NrB = len(B)
    NcB = len(B[0])

    assert NcA == NrB

    res = []

    for i in range(NcB):
        sum = 0
        for j in range(NcA):
            sum += B[j][i] * A[j]
        res.append(sum)
    return(res)

# Scalar product of two arrays with jit decorator
@jit
def CPU_loop_ArrayMulJit(A,B):

    NcA = len(A)
    NrB = len(B)

    assert NcA == NrB

    res = 0
    for i in range(NcA):
        res += A[i]*B[i]
    return(res)

# Multiplies array A per matrix B with jit decorator
@jit
def CPU_loop_ArrayMatMulJit(A,B):

    NcA = len(A)
    NrB = len(B)
    NcB = len(B[0])

    assert NcA == NrB

    res = []

    for i in range(NcB):
        sum = 0
        for j in range(NcA):
            sum += B[j][i] * A[j]
        res.append(sum)
    return(res)

##1. Basic Partition Loop Function
def CPU_loop_Partition(W,b,c,x,h):
    Sum_x_h = 0
    c_transp = np.transpose(c)
    for i in range(2**n): # Sum x
        E_x_h = 0
        x_transp = np.transpose(x[i])
        for j in range(2**m): # Sum h
            E_x_h += CPU_loop_ArrayMul(x_transp,b) + CPU_loop_ArrayMul(c_transp,h[j]) + CPU_loop_ArrayMul(CPU_loop_ArrayMatMul(x_transp,W),h[j])
        Sum_x_h += e**E_x_h
    return(Sum_x_h)

##2. CPU + Numba Partition
@jit
def CPU_loop_numba_Partition(W,b,c,x,h):
    Sum_x_h = 0
    c_transp = np.transpose(c)
    for i in range(2**n): # Sum x
        E_x_h = 0
        x_transp = np.transpose(x[i])
        for j in range(2**m): # Sum h
            E_x_h += CPU_loop_ArrayMulJit(x_transp,b) + CPU_loop_ArrayMulJit(c_transp,h[j]) + CPU_loop_ArrayMulJit(CPU_loop_ArrayMatMulJit(x_transp,W),h[j])
        Sum_x_h += e**E_x_h
    return(Sum_x_h)

@jit(nopython=False, forceobj=True)
def CPU_loop_numba_PartitionObject(W,b,c,x,h):
    Sum_x_h = 0
    c_transp = np.transpose(c)
    for i in range(2**n): # Sum x
        E_x_h = 0
        x_transp = np.transpose(x[i])
        for j in range(2**m): # Sum h
            E_x_h += CPU_loop_ArrayMulJit(x_transp,b) + CPU_loop_ArrayMulJit(c_transp,h[j]) + CPU_loop_ArrayMulJit(CPU_loop_ArrayMatMulJit(x_transp,W),h[j])
        Sum_x_h += e**E_x_h
    return(Sum_x_h)

@jit(nopython=True)
def CPU_loop_numba_PartitionNJit(W,b,c,x,h):
    Sum_x_h = 0
    c_transp = np.transpose(c)
    for i in range(2**n): # Sum x
        E_x_h = 0
        x_transp = np.transpose(x[i])
        for j in range(2**m): # Sum h
            E_x_h += CPU_loop_ArrayMulJit(x_transp,b) + CPU_loop_ArrayMulJit(c_transp,h[j]) + CPU_loop_ArrayMulJit(CPU_loop_ArrayMatMulJit(x_transp,W),h[j])
        Sum_x_h += e**E_x_h
    return(Sum_x_h)

@jit(float64(float64[:,:], float64[:], float64[:], float64[:,:], float64[:,:]))
def CPU_loop_numba_PartitionEager(W,b,c,x,h):
    Sum_x_h = 0
    c_transp = np.transpose(c)
    for i in range(2**n): # Sum x
        E_x_h = 0
        x_transp = np.transpose(x[i])
        for j in range(2**m): # Sum h
            E_x_h += CPU_loop_ArrayMulJit(x_transp,b) + CPU_loop_ArrayMulJit(c_transp,h[j]) + CPU_loop_ArrayMulJit(CPU_loop_ArrayMatMulJit(x_transp,W),h[j])
        Sum_x_h += e**E_x_h
    return(Sum_x_h)

@jit(float64(float64[:,:], float64[:], float64[:], float64[:,:], float64[:,:],float64,float64), nopython=True)
def CPU_loop_numba_PartitionEagerNJit(W,b,c,x,h,n,m):
    Sum_x_h = 0
    c_transp = np.transpose(c)
    for i in range(2**n): # Sum x
        E_x_h = 0
        x_transp = np.transpose(x[i])
        for j in range(2**m): # Sum h
            E_x_h += CPU_loop_ArrayMulJit(x_transp,b) + CPU_loop_ArrayMulJit(c_transp,h[j]) + CPU_loop_ArrayMulJit(CPU_loop_ArrayMatMulJit(x_transp,W),h[j])
        Sum_x_h += e**E_x_h
    return(Sum_x_h)

@jit(nogil=True)
def CPU_loop_numba_PartitionNoGil(W,b,c,x,h):
    Sum_x_h = 0
    c_transp = np.transpose(c)
    for i in range(2**n): # Sum x
        E_x_h = 0
        x_transp = np.transpose(x[i])
        for j in range(2**m): # Sum h
            E_x_h += CPU_loop_ArrayMulJit(x_transp,b) + CPU_loop_ArrayMulJit(c_transp,h[j]) + CPU_loop_ArrayMulJit(CPU_loop_ArrayMatMulJit(x_transp,W),h[j])
        Sum_x_h += e**E_x_h
    return(Sum_x_h)

@jit(cache=True)
def CPU_loop_numba_PartitionCache(W,b,c,x,h):
    Sum_x_h = 0
    c_transp = np.transpose(c)
    for i in range(2**n): # Sum x
        E_x_h = 0
        x_transp = np.transpose(x[i])
        for j in range(2**m): # Sum h
            E_x_h += CPU_loop_ArrayMulJit(x_transp,b) + CPU_loop_ArrayMulJit(c_transp,h[j]) + CPU_loop_ArrayMulJit(CPU_loop_ArrayMatMulJit(x_transp,W),h[j])
        Sum_x_h += e**E_x_h
    return(Sum_x_h)

@jit(parallel=True)
def CPU_loop_numba_PartitionParallel(W,b,c,x,h):
    Sum_x_h = 0
    c_transp = np.transpose(c)
    for i in range(2**n): # Sum x
        E_x_h = 0
        x_transp = np.transpose(x[i])
        for j in range(2**m): # Sum h
            E_x_h += CPU_loop_ArrayMulJit(x_transp,b) + CPU_loop_ArrayMulJit(c_transp,h[j]) + CPU_loop_ArrayMulJit(CPU_loop_ArrayMatMulJit(x_transp,W),h[j])
        Sum_x_h += e**E_x_h
    return(Sum_x_h)

# @guvectorize([float64[:,:], float64[:], float64[:], float64[:,:], float64[:,:], float64], '(n,m),(m,p)->(n,p)', nopython=True)
# def CPU_loop_numba_PartitionGuVectorize(W,b,c,x,h,Sum_x_h):
#     c_transp = np.transpose(c)
#     for i in range(2**n): # Sum x
#         E_x_h = 0
#         x_transp = np.transpose(x[i])
#         for j in range(2**m): # Sum h
#             E_x_h += CPU_loop_ArrayMulJit(x_transp,b) + CPU_loop_ArrayMulJit(c_transp,h[j]) + CPU_loop_ArrayMulJit(CPU_loop_ArrayMatMulJit(x_transp,W),h[j])
#         Sum_x_h += e**E_x_h

@cfunc("float64(float64[:,:], float64[:], float64[:], float64[:,:], float64[:,:], float64, float64)")
def CPU_loop_numba_PartitionCFunc(W,b,c,x,h,n,m):
    Sum_x_h = 0
    c_transp = np.transpose(c)
    for i in range(2**n): # Sum x
        E_x_h = 0
        x_transp = np.transpose(x[i])
        for j in range(2**m): # Sum h
            E_x_h += CPU_loop_ArrayMulJit(x_transp,b) + CPU_loop_ArrayMulJit(c_transp,h[j]) + CPU_loop_ArrayMulJit(CPU_loop_ArrayMatMulJit(x_transp,W),h[j])
        Sum_x_h += e**E_x_h
    return(Sum_x_h)

@jit(fastmath = True)
def CPU_loop_numba_PartitionFastMath(W,b,c,x,h):
    Sum_x_h = 0
    c_transp = np.transpose(c)
    for i in range(2**n): # Sum x
        E_x_h = 0
        x_transp = np.transpose(x[i])
        for j in range(2**m): # Sum h
            E_x_h += CPU_loop_ArrayMulJit(x_transp,b) + CPU_loop_ArrayMulJit(c_transp,h[j]) + CPU_loop_ArrayMulJit(CPU_loop_ArrayMatMulJit(x_transp,W),h[j])
        Sum_x_h += e**E_x_h
    return(Sum_x_h)

##3. Numpy partition
def CPU_numpy_Partition(W,b,c,x,h):
    Sum_x_h = 0
    c_transp = np.transpose(c)
    for i in range(2**n): # Sum x
        E_x_h = 0
        x_transp = np.transpose(x[i])
        for j in range(2**m): # Sum h
            E_x_h += np.dot(x_transp,b) + np.dot(c_transp,h[j]) + np.dot(np.dot(x_transp,W),h[j])
        Sum_x_h += e**E_x_h
    return(Sum_x_h)

##4. Numpy + numba partition
@jit(nopython=False)
def CPU_numpy_numba_Partition(W,b,c,x,h):
    Sum_x_h = 0
    c_transp = np.transpose(c)
    for i in range(2**n): # Sum x
        E_x_h = 0
        x_transp = np.transpose(x[i])
        for j in range(2**m): # Sum h
            E_x_h += np.dot(x_transp,b) + np.dot(c_transp,h[j]) + np.dot(np.dot(x_transp,W),h[j])
        Sum_x_h += e**E_x_h
    return(Sum_x_h)

# # Additional function for transpose matrix (This cannot be done directly in a Numba function because it is not supported by numba)
# @cupy.fuse()
# def cupy_transpose(a):
#     return a.transpose()

# #5. GPU loop partition
# @cuda.jit
# def GPU_loop_numba_Partition(W,b,c,x,h):
#     Sum_x_h = 0
#     c_transp = cupy_transpose(c)
#     i = cuda.grid(1)
#     j = cuda.grid(1)
#     for i_ in range(2**n): #Sum x
#         E_x_h = 0
#         x_transp = cupy_transpose(x[i])
#         for j_ in range(2**m): # Sum h
#             E_x_h += cp.dot(x_transp,b) + cp.dot(c_transp,h[j]) + cp.dot(cp.dot(x_transp,W),h[j])
#         Sum_x_h += e**E_x_h
#     return(Sum_x_h)

##5. GPU loop partition
@cuda.jit
def GPU_loop_Partition(W, b, c, x, h, result):
    idx = cuda.grid(1)
    if idx < x.shape[0]:
        E_x_h = 0.0
        for j in range(2**m):
            dot_x_b = 0.0
            for k in range(n):
                dot_x_b += x[idx, k] * b[k]
            dot_x_W_h = 0.0
            for k in range(m):
                dot_x_W_h_k = 0.0
                for l in range(n):
                    dot_x_W_h_k += x[idx, l] * W[l, k]
                dot_x_W_h += dot_x_W_h_k * h[j, k]
            E_x_h += dot_x_b + dot_x_W_h + c[j]
        result[idx] = e**(E_x_h)

# def GPU_loop_Partition(W, b, c, x, h):
#     threads_per_block = 2
#     blocks_per_grid = 4
#     partition_kernel[blocks_per_grid, threads_per_block](W, b, c, x, h, result)
#     return np.sum(result)

##########################################################################
################################## MAIN ##################################
##########################################################################

#print(CPU_loop_Partition())

print("====================================================")

# Partition Vars:
n = 10
m = 10

R = 30

W = np.random.random((n,m)) * 0.0000001 # Matrix size NxM, small values
b = np.random.random((n)) * 0.0000001 # Array size N, small values
c = np.random.random((m)) * 0.0000001 # Array size M, small values

# Create array x, size 2^n
x = np.zeros((2**n, n), dtype=np.float64)
for i in range(2**n):
    x_i_bin = np.binary_repr(i, n)
    x[i] = [int(j) for j in str(x_i_bin)]

# Create array h, size 2^m
h = np.zeros((2**m, m), dtype=np.float64)
for i in range(2**m):
    h_i_bin = np.binary_repr(i, m)
    h[i] = [int(j) for j in str(h_i_bin)]

threads_per_block = 64
blocks_per_grid = 16

print("Threads per block:", threads_per_block)
print("Blocks per grid:", blocks_per_grid)

# print("W:", W)
# print("b:", b)
# print("c:", c)
# print("x:", x)
# print("h:", h)

x_comp = ["CPU - numpy + numba", "CPU - loop + numba (Jit)",
     # "CPU - loop + numba (Object)", 
     "CPU - loop + numba (NJit)",
     #"CPU - loop + numba (Eager)", 
     "CPU - loop + numba (Eager, NJit)",
     "CPU - loop + numba (NoGil)", "CPU - loop + numba (Cache)",
     #"CPU - loop + numba (Parallel)", "CPU - loop + numba (GuVectorize)",
     #"CPU - loop + numba (CFunc)", 
     "CPU - loop + numba (FastMath)",
     "GPU - loop + numba"]
x_exec = [#"CPU - Loop", 
     "CPU - numpy", "CPU - numpy + numba", "CPU - loop + numba (Jit)",
     #"CPU - loop + numba (Object)", 
     "CPU - loop + numba (NJit)",
     #"CPU - loop + numba (Eager)", 
     "CPU - loop + numba (Eager, NJit)",
     "CPU - loop + numba (NoGil)", "CPU - loop + numba (Cache)",
     #"CPU - loop + numba (Parallel)", "CPU - loop + numba (GuVectorize)",
     # "CPU - loop + numba (CFunc)", 
     "CPU - loop + numba (FastMath)",
     "GPU - loop + numba"]
y_comp = []
y_exec = []

# ### CPU - loop
# tic = time()
# for i in range(R):
#     res = CPU_loop_Partition(W,b,c,x,h)

# exec = time() - tic
# y_exec.append(exec)


# print(" Partition - CPU - loop:         {}", exec)
# print(res)

### CPU - numpy
tic = time()
for i in range(R):
    res = CPU_numpy_Partition(W,b,c,x,h)

exec = time() - tic
y_exec.append(exec)


print(" Partition - CPU - numpy:          {}", exec)
print(res)
# print(np.dtype(res))

### CPU - numpy + numba
tic = time()
res = CPU_numpy_numba_Partition(W,b,c,x,h)   ### Compilation
tic2 = time()
for i in range(R):
    res = CPU_numpy_numba_Partition(W,b,c,x,h)

exec = time() - tic2
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)

print(" Partition - CPU - numpy + numba (exec):  {}", exec)
print(" Partition - CPU - numpy + numba:  {}", comp)
print(res)

################ Numba ######################

### CPU - numba (jit)
tic = time()
res = CPU_loop_numba_Partition(W,b,c,x,h)  ### Compilation
tic2 = time()
for i in range(R):
    res = CPU_loop_numba_Partition(W,b,c,x,h)

exec = time() - tic2
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)

print(" Partition - CPU - numba (jit) (exec): {}", exec)
print(" Partition - CPU - numba (jit): {}", comp)
print(res)

# # CPU - loop + numba (Object Mode)
# tic = time()
# res = CPU_loop_numba_PartitionObject(W,b,c,x,h)   ### Compilation
# tic2 = time()
# for i in range(R):
#     res = CPU_loop_numba_PartitionObject(W,b,c,x,h)

# exec = time() - tic2
# comp = time() - tic

# y_exec.append(exec)
# y_comp.append(comp)

# print(" Partition - CPU - loop + numba (Object) (exec):  {}", exec)
# print(" Partition - CPU - loop + numba (Object):  {}", comp)
# print(res)

# CPU - loop + numba (njit)
tic = time()
res = CPU_loop_numba_PartitionNJit(W,b,c,x,h)    ### Compilation
tic2 = time()
for i in range(R):
    res = CPU_loop_numba_PartitionNJit(W,b,c,x,h)

exec = time() - tic2
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)

print(" Partition - CPU - loop + numba (njit) (exec):  {}", exec)
print(" Partition - CPU - loop + numba (njit):  {}", comp)
print(res)

# # CPU - loop + numba (Eager)
# tic = time()
# res = CPU_loop_numba_PartitionEager(W,b,c,x,h)    ### Compilation
# tic2 = time()
# for i in range(R):
#     res = CPU_loop_numba_PartitionEager(W,b,c,x,h)

# exec = time() - tic2
# comp = time() - tic

# y_exec.append(exec)
# y_comp.append(comp)

# print(" Partition - CPU - loop + numba (Eager) (exec):  {}", exec)
# print(" Partition - CPU - loop + numba (Eager):  {}", comp)
# print(res)

# CPU - loop + numba (Eager, NJit)
tic = time()
res = CPU_loop_numba_PartitionEagerNJit(W,b,c,x,h,n,m)   ### Compilation
tic2 = time()
for i in range(R):
    res = CPU_loop_numba_PartitionEagerNJit(W,b,c,x,h,n,m)

exec = time() - tic2
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)

print(" Partition - CPU - loop + numba (Eager, NJit) (exec):  {}", exec)
print(" Partition - CPU - loop + numba (Eager, NJit):  {}", comp)
print(res)

# CPU - loop + numba (NoGil)
tic = time()
res = CPU_loop_numba_PartitionNoGil(W,b,c,x,h)   ### Compilation
tic2 = time()
for i in range(R):
    res = CPU_loop_numba_PartitionNoGil(W,b,c,x,h)

exec = time() - tic2
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)

print(" Partition - CPU - loop + numba (NoGil) (exec):  {}", exec)
print(" Partition - CPU - loop + numba (NoGil):  {}", comp)
print(res)

# CPU - loop + numba (Cache)
tic = time()
res = CPU_loop_numba_PartitionCache(W,b,c,x,h)   ### Compilation
tic2 = time()
for i in range(R):
    res = CPU_loop_numba_PartitionCache(W,b,c,x,h)

exec = time() - tic2
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)

print(" Partition - CPU - loop + numba (Cache) (exec):  {}", exec)
print(" Partition - CPU - loop + numba (Cache):  {}", comp)
print(res)

# # CPU - loop + numba (CFunc)
# tic = time()
# res = CPU_loop_numba_PartitionCFunc(W,b,c,x,h,n,m)   ### Compilation
# tic2 = time()
# for i in range(R):
#     res = CPU_loop_numba_PartitionCFunc(W,b,c,x,h,n,m)

# exec = time() - tic2
# comp = time() - tic

# y_exec.append(exec)
# y_comp.append(comp)

# print(" Partition - CPU - loop + numba (CFunc) (exec):  {}", exec)
# print(" Partition - CPU - loop + numba (CFunc):  {}", comp)
# print(res)

# CPU - loop + numba (FastMath)
tic = time()
res = CPU_loop_numba_PartitionFastMath(W,b,c,x,h)   ### Compilation
tic2 = time()
for i in range(R):
    res = CPU_loop_numba_PartitionFastMath(W,b,c,x,h)

exec = time() - tic2
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)

print(" Partition - CPU - loop + numba (FastMath) (exec):  {}", exec)
print(" Partition - CPU - loop + numba (FastMath):  {}", comp)
print(res)


#############################################

### GPU - loop
result = np.zeros(x.shape[0], dtype=np.float64)
cW = cuda.to_device(W)
cb = cuda.to_device(b)
cc = cuda.to_device(c)
cx = cuda.to_device(x)
ch = cuda.to_device(h)
cresult = cuda.to_device(result)

# print(np.dtype(W[0][0]))
# print(np.dtype(b[0]))
# print(np.dtype(c[0]))
# print(np.dtype(x[0][0]))
# print(np.dtype(h[0][0]))

# print(np.dtype(cW))
# print(np.dtype(cb))
# print(np.dtype(cc))
# print(np.dtype(cx))
# print(np.dtype(ch))
# print(np.dtype(cresult))

tic = time()
GPU_loop_Partition[blocks_per_grid, threads_per_block](cW,cb,cc,cx,ch, cresult)
tic2 = time()
for i in range(R):
    GPU_loop_Partition[blocks_per_grid, threads_per_block](cW,cb,cc,cx,ch, cresult)
    res = np.sum(cresult)

exec = time() - tic2
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)

print(" Partition - GPU - loop (exec):  {}", exec)
print(" Partition - GPU - loop:  {}", comp)
print(res)
# print(np.dtype(res))

# Plot printing:

# Convert the results to miliseconds:
for i in y_exec:
  i = i*1000

plt.figure(figsize=(10, 6))
plt.vlines(x_exec, 0, y_exec, linestyle="dashed", colors="red")
plt.scatter(x_exec, y_exec, c="red")
plt.xticks(x_exec, rotation=45, ha='right')
plt.title("Tiempos de ejecución - Partición N = 10, M = 10", pad=20)
plt.yscale('log')
plt.ylabel("Tiempo en ms", labelpad=20)
plt.xlabel("Implementación", labelpad=10)
plt.show()

# Execution and compilation time plot:

# Convert the results to miliseconds:
for i in y_comp:
  i = i*1000

plt.figure(figsize=(10, 6))
plt.vlines(x_comp, 0, y_comp, linestyle="dashed", colors="red")
plt.scatter(x_comp, y_comp, c="red")
plt.xticks(x_comp, rotation=45, ha='right')
plt.title("Tiempos de compilación y ejecución - Partición N = 10, M = 10", pad=20)
plt.yscale('log')
plt.ylabel("Tiempo en ms", labelpad=20)
plt.xlabel("Implementación", labelpad=10)
plt.show()

print(len(y_exec))
print(len(y_comp))
