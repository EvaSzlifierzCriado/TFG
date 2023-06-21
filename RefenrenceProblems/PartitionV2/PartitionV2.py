import numpy as np
from math import e
import numba.cuda as cuda
from time import time
import numba
from numba import jit, cuda, float64, guvectorize, cfunc
import math
import matplotlib.pyplot as plt

# PartitionV2 Vars:
n = 12
m = 12

R = 30

W = np.random.random((n,m)) # Matrix size NxM, small values
b = np.random.random((n)) # Array size N, small values
c = np.random.random((m)) # Array size M, small values

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

BIN = np.zeros((2), dtype=np.float64)
BIN[1] = 1.0   ### Possible values of every position (only [0,1] or [-1,+1])

threads_per_block = 16
blocks_per_grid = 256

print("Threads per block:", threads_per_block)
print("Blocks per grid", blocks_per_grid)

########################################
######## Plot vars: ####################
########################################

x_comp = [#"CPU - numpy + numba", 
     "CPU - loop + numba (Jit)",
     "CPU - loop + numba (Object)", "CPU - loop + numba (NJit)", 
     "CPU - loop + numba (Eager)", "CPU - loop + numba (Eager, NJit)",
     "CPU - loop + numba (NoGil)", "CPU - loop + numba (Cache)",
     "CPU - loop + numba (CFunc)", "CPU - loop + numba (FastMath)",
     "GPU - loop + numba"]
x_exec = ["CPU - Loop", "CPU - numpy", #"CPU - numpy + numba", 
     "CPU - loop + numba (Jit)",
     "CPU - loop + numba (Object)", "CPU - loop + numba (NJit)", 
     "CPU - loop + numba (Eager)", "CPU - loop + numba (Eager, NJit)",
     "CPU - loop + numba (NoGil)", "CPU - loop + numba (Cache)",
     "CPU - loop + numba (CFunc)", "CPU - loop + numba (FastMath)",
     "GPU - loop + numba"]
y_comp = []
y_exec = []

# print("W:", W)
# print("b:", b)
# print("c:", c)
# print("x:", x)
# print("h:", h)

###############################################

#V2
def CPU_loop_PartitionV2(W,b,c,x,h):
    Prod_h_x = 0
    c_transp = np.transpose(c)
    for j in range(2**m): # Sum h 
        Prod_x = 1
        for i in range(n): # Product of f_i(a_i) + f_i(b_i) N times, being N the number of digits
            sum_x = 0
            for k in range(2):
                sum_x += e**(BIN[k]*(CPU_loop_ArrayMul(W[i],h[j]) + b[i]))
            Prod_x *= sum_x
        Prod_h_x += ((e**(CPU_loop_ArrayMul(c_transp, h[j]))) * Prod_x)
    return(Prod_h_x)

###############################################

# Scalar product of two arrays
def CPU_loop_ArrayMul(A,B):

    NcA = len(A)
    NrB = len(B)

    assert NcA == NrB

    res = 0
    for i in range(NcA):
        res += A[i]*B[i]
    return(res)

# # Multiplies array A per matrix B
# def CPU_loop_ArrayMatMul(A,B): 

#     NcA = len(A)
#     NrB = len(B)
#     NcB = len(B[0])

#     assert NcA == NrB

#     res = []

#     for i in range(NcB):
#         sum = 0
#         for j in range(NcA):
#             sum += A[j] * B[j][i]
#         res.append(sum)
#     return(res)

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

# # Multiplies array A per matrix B with jit decorator
# def CPU_loop_ArrayMatMulJit(A,B): 

#     NcA = len(A)
#     NrB = len(B)
#     NcB = len(B[0])

#     assert NcA == NrB

#     res = []

#     for i in range(NcB):
#         sum = 0
#         for j in range(NcA):
#             sum += B[j][i] * A[j]
#         res.append(sum)
#     return(res)

# # V1:
# def CPU_loop_PartitionV2(W, b, c, x, h): 
#     sum_xh = 0
#     for i in range(2**n):
#         for j in range(2**m):
#             sum_xh += e**(CPU_loop_ArrayMul(np.transpose(b), x[i]) + CPU_loop_ArrayMul(np.transpose(c), h[j]) + CPU_loop_ArrayMul(CPU_loop_ArrayMatMul(np.transpose(x[i]), W), h[j]))
#     return sum_xh


##2. CPU + Numba PartitionV2
@jit
def CPU_loop_numba_PartitionV2(W,b,c,x,h):
    Prod_h_x = 0
    c_transp = np.transpose(c)
    for j in range(2**m): # Sum h
        Prod_x = 1
        for i in range(n):
            sum_x = 0
            for k in range(2):
                sum_x += e**(BIN[k]*(CPU_loop_ArrayMulJit(W[i],h[j]) + b[i]))
            Prod_x *= sum_x
        Prod_h_x += ((e**(CPU_loop_ArrayMulJit(c_transp, h[j]))) * Prod_x)
    return(Prod_h_x)

@jit(nopython=False, forceobj=True)
def CPU_loop_numba_PartitionV2Object(W,b,c,x,h):
    Prod_h_x = 0
    c_transp = np.transpose(c)
    for j in range(2**m): # Sum h
        Prod_x = 1
        for i in range(n):
            sum_x = 0
            for k in range(2):
                sum_x += e**(BIN[k]*(CPU_loop_ArrayMulJit(W[i],h[j]) + b[i]))
            Prod_x *= sum_x
        Prod_h_x += ((e**(CPU_loop_ArrayMulJit(c_transp, h[j]))) * Prod_x)
    return(Prod_h_x)

@jit(nopython=True)
def CPU_loop_numba_PartitionV2NJit(W,b,c,x,h):
    Prod_h_x = 0
    c_transp = np.transpose(c)
    for j in range(2**m): # Sum h
        Prod_x = 1
        for i in range(n):
            sum_x = 0
            for k in range(2):
                sum_x += e**(BIN[k]*(CPU_loop_ArrayMulJit(W[i],h[j]) + b[i]))
            Prod_x *= sum_x
        Prod_h_x += ((e**(CPU_loop_ArrayMulJit(c_transp, h[j]))) * Prod_x)
    return(Prod_h_x)

@jit(float64(float64[:,:], float64[:], float64[:], float64[:,:], float64[:,:]))
def CPU_loop_numba_PartitionV2Eager(W,b,c,x,h):
    Prod_h_x = 0
    c_transp = np.transpose(c)
    for j in range(2**m): # Sum h
        Prod_x = 1
        for i in range(n):
            sum_x = 0
            for k in range(2):
                sum_x += e**(BIN[k]*(CPU_loop_ArrayMulJit(W[i],h[j]) + b[i]))
            Prod_x *= sum_x
        Prod_h_x += ((e**(CPU_loop_ArrayMulJit(c_transp, h[j]))) * Prod_x)
    return(Prod_h_x)

@jit(float64(float64[:,:], float64[:], float64[:], float64[:,:], float64[:,:],float64,float64), nopython=True)
def CPU_loop_numba_PartitionV2EagerNJit(W,b,c,x,h,n,m):
    Prod_h_x = 0
    c_transp = np.transpose(c)
    for j in range(2**m): # Sum h
        Prod_x = 1
        for i in range(n):
            sum_x = 0
            for k in range(2):
                sum_x += e**(BIN[k]*(CPU_loop_ArrayMulJit(W[i],h[j]) + b[i]))
            Prod_x *= sum_x
        Prod_h_x += ((e**(CPU_loop_ArrayMulJit(c_transp, h[j]))) * Prod_x)
    return(Prod_h_x)

@jit(nogil=True)
def CPU_loop_numba_PartitionV2NoGil(W,b,c,x,h):
    Prod_h_x = 0
    c_transp = np.transpose(c)
    for j in range(2**m): # Sum h
        Prod_x = 1
        for i in range(n):
            sum_x = 0
            for k in range(2):
                sum_x += e**(BIN[k]*(CPU_loop_ArrayMulJit(W[i],h[j]) + b[i]))
            Prod_x *= sum_x
        Prod_h_x += ((e**(CPU_loop_ArrayMulJit(c_transp, h[j]))) * Prod_x)
    return(Prod_h_x)

@jit(cache=True)
def CPU_loop_numba_PartitionV2Cache(W,b,c,x,h):
    Prod_h_x = 0
    c_transp = np.transpose(c)
    for j in range(2**m): # Sum h
        Prod_x = 1
        for i in range(n):
            sum_x = 0
            for k in range(2):
                sum_x += e**(BIN[k]*(CPU_loop_ArrayMulJit(W[i],h[j]) + b[i]))
            Prod_x *= sum_x
        Prod_h_x += ((e**(CPU_loop_ArrayMulJit(c_transp, h[j]))) * Prod_x)
    return(Prod_h_x)

@jit(parallel=True)
def CPU_loop_numba_PartitionV2Parallel(W,b,c,x,h):
    Prod_h_x = 0
    c_transp = np.transpose(c)
    for j in range(2**m): # Sum h
        Prod_x = 1
        for i in range(n):
            sum_x = 0
            for k in range(2):
                sum_x += e**(BIN[k]*(CPU_loop_ArrayMulJit(W[i],h[j]) + b[i]))
            Prod_x *= sum_x
        Prod_h_x += ((e**(CPU_loop_ArrayMulJit(c_transp, h[j]))) * Prod_x)
    return(Prod_h_x)

# @guvectorize([float64[:,:], float64[:], float64[:], float64[:,:], float64[:,:], float64], '(n,m),(m,p)->(n,p)', nopython=True)
# def CPU_loop_numba_PartitionV2GuVectorize(W,b,c,x,h,Sum_x_h):
#     c_transp = np.transpose(c)
#     for i in range(2**n): # Sum x
#         E_x_h = 0
#         x_transp = np.transpose(x[i])
#         for j in range(2**m): # Sum h
#             E_x_h += CPU_loop_ArrayMulJit(x_transp,b) + CPU_loop_ArrayMulJit(c_transp,h[j]) + CPU_loop_ArrayMulJit(CPU_loop_ArrayMatMulJit(x_transp,W),h[j])
#         Sum_x_h += e**E_x_h

@cfunc("float64(float64[:,:], float64[:], float64[:], float64[:,:], float64[:,:], float64, float64)")
def CPU_loop_numba_PartitionV2CFunc(W,b,c,x,h,n,m):
    Prod_h_x = 0
    c_transp = np.transpose(c)
    for j in range(2**m): # Sum h
        Prod_x = 1
        for i in range(n):
            sum_x = 0
            for k in range(2):
                sum_x += e**(BIN[k]*(CPU_loop_ArrayMulJit(W[i],h[j]) + b[i]))
            Prod_x *= sum_x
        Prod_h_x += ((e**(CPU_loop_ArrayMulJit(c_transp, h[j]))) * Prod_x)
    return(Prod_h_x)

@jit(fastmath = True)
def CPU_loop_numba_PartitionV2FastMath(W,b,c,x,h):
    Prod_h_x = 0
    c_transp = np.transpose(c)
    for j in range(2**m): # Sum h
        Prod_x = 1
        for i in range(n):
            sum_x = 0
            for k in range(2):
                sum_x += e**(BIN[k]*(CPU_loop_ArrayMulJit(W[i],h[j]) + b[i]))
            Prod_x *= sum_x
        Prod_h_x += ((e**(CPU_loop_ArrayMulJit(c_transp, h[j]))) * Prod_x)
    return(Prod_h_x)

##3. Numpy PartitionV2
def CPU_numpy_PartitionV2(W,b,c,x,h):
    Prod_h_x = 0
    c_transp = np.transpose(c)
    for j in range(2**m): # Sum h 
        Prod_x = 1
        for i in range(n): 
            sum_x = 0
            for k in range(2):
                WH = np.dot(W[i],h[j])
                sum_x += e**(np.dot(BIN[k],(WH + b[i])))
            Prod_x *= sum_x
        Prod_h_x += e**(np.dot(c_transp,h[j])) * Prod_x
    return(Prod_h_x)

##4. Numpy + numba PartitionV2
@jit(nopython=False)
def CPU_numpy_numba_PartitionV2(W,b,c,x,h):
    Prod_h_x = 0
    c_transp = np.transpose(c)
    for j in range(2**m): # Sum h 
        Prod_x = 1
        for i in range(n): 
            sum_x = 0
            for k in range(2):
                WH = np.dot(W[i],h[j])
                sum_x += e**(np.dot(BIN[k],(WH + b[i])))
            Prod_x *= sum_x
        Prod_h_x += e**(np.dot(c_transp,h[j])) * Prod_x
    return(Prod_h_x)

@cuda.jit
def GPU_loop_PartitionV2(W, b, c, x, h, cBIN, result):
    j = cuda.grid(1) # 1D grid
    if j < 2**m:
        Prod_x = 1 
        for i in range(n): 
            sum_x = 0
            WH = 0
            for l in range(m):
                WH += W[i][l]*h[j][l]
            for k in range(2):
                sum_x += e**(cBIN[k]*(WH + b[i]))
            Prod_x *= sum_x
        CH = 0
        for l in range(m):
            CH += c[l]*h[j][l]
        result[j] = e**CH * Prod_x

### CPU - loop
tic = time()
for i in range(R):
    res = CPU_loop_PartitionV2(W,b,c,x,h)

exec = time() - tic
y_exec.append(exec)

print(" PartitionV2 - CPU - loop:         {}", exec)
print(res)

### CPU - numpy
tic = time()
for i in range(R):
    res = CPU_numpy_PartitionV2(W,b,c,x,h)

exec = time() - tic
y_exec.append(exec)

print(" PartitionV2 - CPU - numpy:          {}", exec)
print(res)

################ Numba ######################


### CPU - numba (jit)
tic = time()
res = CPU_loop_numba_PartitionV2(W,b,c,x,h)  ### Compilation
tic2 = time()
for i in range(R):
    res = CPU_loop_numba_PartitionV2(W,b,c,x,h)

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)

print(" PartitionV2 - CPU - numba (jit) (exec): {}", exec)
print(" PartitionV2 - CPU - numba (jit): {}", comp)
print(res)

# CPU - loop + numba (Object Mode)
tic = time()
res = CPU_loop_numba_PartitionV2Object(W,b,c,x,h)   ### Compilation
tic2 = time()
for i in range(R):
    res = CPU_loop_numba_PartitionV2Object(W,b,c,x,h) 

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)

print(" PartitionV2 - CPU - loop + numba (Object) (exec):  {}", exec)
print(" PartitionV2 - CPU - loop + numba (Object):  {}", comp)
print(res)

# CPU - loop + numba (njit)
tic = time()
res = CPU_loop_numba_PartitionV2NJit(W,b,c,x,h)    ### Compilation
tic2 = time()
for i in range(R):
    res = CPU_loop_numba_PartitionV2NJit(W,b,c,x,h) 

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)

print(" PartitionV2 - CPU - loop + numba (njit) (exec):  {}", exec)
print(" PartitionV2 - CPU - loop + numba (njit):  {}", comp)
print(res)

# CPU - loop + numba (Eager)
tic = time()
res = CPU_loop_numba_PartitionV2Eager(W,b,c,x,h)    ### Compilation
tic2 = time()
for i in range(R):
    res = CPU_loop_numba_PartitionV2Eager(W,b,c,x,h) 

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)

print(" PartitionV2 - CPU - loop + numba (Eager) (exec):  {}", exec)
print(" PartitionV2 - CPU - loop + numba (Eager):  {}", comp)
print(res)

# CPU - loop + numba (Eager, NJit)
tic = time()
res = CPU_loop_numba_PartitionV2EagerNJit(W,b,c,x,h,n,m)   ### Compilation
tic2 = time()
for i in range(R):
    res = CPU_loop_numba_PartitionV2EagerNJit(W,b,c,x,h,n,m)

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)

print(" PartitionV2 - CPU - loop + numba (Eager, NJit) (exec):  {}", exec)
print(" PartitionV2 - CPU - loop + numba (Eager, NJit):  {}", comp)
print(res)

# CPU - loop + numba (NoGil)
tic = time()
res = CPU_loop_numba_PartitionV2NoGil(W,b,c,x,h)   ### Compilation
tic2 = time()
for i in range(R):
    res = CPU_loop_numba_PartitionV2NoGil(W,b,c,x,h)

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)

print(" PartitionV2 - CPU - loop + numba (NoGil) (exec):  {}", exec)
print(" PartitionV2 - CPU - loop + numba (NoGil):  {}", comp)
print(res)

# CPU - loop + numba (Cache)
tic = time()
res = CPU_loop_numba_PartitionV2Cache(W,b,c,x,h)   ### Compilation
tic2 = time()
for i in range(R):
    res = CPU_loop_numba_PartitionV2Cache(W,b,c,x,h)

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)

print(" PartitionV2 - CPU - loop + numba (Cache) (exec):  {}", exec)
print(" PartitionV2 - CPU - loop + numba (Cache):  {}", comp)
print(res)

# CPU - loop + numba (CFunc)
tic = time()
res = CPU_loop_numba_PartitionV2CFunc(W,b,c,x,h,n,m)   ### Compilation
tic2 = time()
for i in range(R):
    res = CPU_loop_numba_PartitionV2CFunc(W,b,c,x,h,n,m)

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)

print(" PartitionV2 - CPU - loop + numba (CFunc) (exec):  {}", exec)
print(" PartitionV2 - CPU - loop + numba (CFunc):  {}", comp)
print(res)

# CPU - loop + numba (FastMath)
tic = time()
res = CPU_loop_numba_PartitionV2FastMath(W,b,c,x,h)   ### Compilation
tic2 = time()
for i in range(R):
    res = CPU_loop_numba_PartitionV2FastMath(W,b,c,x,h)

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)

print(" PartitionV2 - CPU - loop + numba (FastMath) (exec):  {}", exec)
print(" PartitionV2 - CPU - loop + numba (FastMath):  {}", comp)
print(res)


#############################################

# ### CPU - numpy + numba
# res = CPU_numpy_numba_PartitionV2(W,b,c,x,h)   ### Compilation
# tic = time()
# for i in range(R):
#     res = CPU_numpy_numba_PartitionV2(W,b,c,x,h)
# print(" PartitionV2 - CPU - numpy + numba:  {}".format(time() - tic))
# print(res)

### GPU - loop
result = np.zeros(h.shape[0], dtype=np.float64)

cW = cuda.to_device(W)
cb = cuda.to_device(b)
cc = cuda.to_device(np.transpose(c))
cx = cuda.to_device(x)
ch = cuda.to_device(h)
cresult = cuda.to_device(result)
cBIN = cuda.to_device(BIN)

# print(np.dtype(cW[0][0]))
# print(np.dtype(cb[0]))
# print(np.dtype(cc[0]))
# print(np.dtype(cx[0][0]))
# print(np.dtype(ch[0][0]))
# print(np.dtype(cresult[0]))
# print(np.dtype(cBIN[0]))

tic = time()
GPU_loop_PartitionV2[blocks_per_grid, threads_per_block](cW,cb,cc,cx,ch,cBIN,cresult)
tic2 = time()
for i in range(R):
    GPU_loop_PartitionV2[blocks_per_grid, threads_per_block](cW,cb,cc,cx,ch,cBIN,cresult)
    res = np.sum(cresult)

exec = time() - tic2 
comp = time() - tic

y_exec.append(exec)
y_comp.append(comp)

print(" PartitionV2 - GPU - loop (exec):  {}", exec)
print(" PartitionV2 - GPU - loop:  {}", comp)
print(res)

# Plot printing:

# Convert the results to miliseconds:
for i in y_exec:
  i = i*1000

plt.figure(figsize=(10, 6))
plt.vlines(x_exec, 0, y_exec, linestyle="dashed", colors="orange")
plt.scatter(x_exec, y_exec, c="orange")
plt.xticks(x_exec, rotation=45, ha='right')
plt.title("Tiempos de ejecución - Partición V2 N = 12, M = 12", pad=20)
plt.yscale('log')
plt.ylabel("Tiempo en ms", labelpad=20)
plt.xlabel("Implementación", labelpad=10)
plt.show()

# Execution and compilation time plot:

# Convert the results to miliseconds:
for i in y_comp:
  i = i*1000

plt.figure(figsize=(10, 6))
plt.vlines(x_comp, 0, y_comp, linestyle="dashed", colors="orange")
plt.scatter(x_comp, y_comp, c="orange")
plt.xticks(x_comp, rotation=45, ha='right')
plt.title("Tiempos de compilación y ejecución - Partición V2 N = 12, M = 12", pad=20)
plt.yscale('log')
plt.ylabel("Tiempo en ms", labelpad=20)
plt.xlabel("Implementación", labelpad=10)
plt.show()

print(len(y_exec))
print(len(y_comp))
