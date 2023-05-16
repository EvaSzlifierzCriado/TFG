import numpy as np
from math import e
import numba.cuda as cuda
from time import time
import numba
from numba import jit, cuda, float64  ### njit, vectorization

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
@jit(nopython=False, forceobj=True)
def CPU_loop_numba_Partition(W,b,c,x,h):
    Sum_x_h = 0
    c_transp = np.transpose(c)
    for i in range(2**n): # Sum x
        E_x_h = 0
        x_transp = np.transpose(x[i])
        for j in range(2**m): # Sum h
            E_x_h += CPU_loop_ArrayMul(x_transp,b) + CPU_loop_ArrayMul(c_transp,h[j]) + CPU_loop_ArrayMul(CPU_loop_ArrayMatMul(x_transp,W),h[j])
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
n = 6
m = 4

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

# print("W:", W)
# print("b:", b)
# print("c:", c)
# print("x:", x)
# print("h:", h)

### CPU - loop
tic = time()
for i in range(R):
    res = CPU_loop_Partition(W,b,c,x,h)
print(" Partition - CPU - loop:         {}".format(time() - tic))
print(res)

### CPU - numba
res = CPU_loop_numba_Partition(W,b,c,x,h)  ### Compilation
tic = time()
for i in range(R):
    res = CPU_loop_numba_Partition(W,b,c,x,h)
print(" Partition - CPU - numba: {}".format(time() - tic))
print(res)

### CPU - numpy
tic = time()
for i in range(R):
    res = CPU_numpy_Partition(W,b,c,x,h)
print(" Partition - CPU - numpy:          {}".format(time() - tic))
print(res)
# print(np.dtype(res))

### CPU - numpy + numba
res = CPU_numpy_numba_Partition(W,b,c,x,h)   ### Compilation
tic = time()
for i in range(R):
    res = CPU_numpy_numba_Partition(W,b,c,x,h)
print(" Partition - CPU - numpy + numba:  {}".format(time() - tic))
print(res)

### GPU - loop
result = np.zeros(x.shape[0], dtype=np.float64)
threads_per_block = 64
blocks_per_grid = 16
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
for i in range(R):
    GPU_loop_Partition[blocks_per_grid, threads_per_block](cW,cb,cc,cx,ch, cresult)
    res = np.sum(cresult)
print(" Partition - GPU - loop:  {}".format(time() - tic))
print(res)
# print(np.dtype(res))


############### Small values ###############:

#  Partition - CPU - loop:         0.008224248886108398
# 36.25802067986214
#  Partition - CPU - numba: 0.003132343292236328
# 36.25802067986214
#  Partition - CPU - numpy:          0.0028841495513916016
# 36.25802067986214
#  Partition - CPU - numpy + numba:  0.0020291805267333984
# 36.25802067986214
#  Partition - GPU - loop:  0.3867149353027344
# 36.258022


################## Bigger values ###############:

#  Partition - CPU - loop:         1.00943922996521
# 4.769582996848387e+74
#  Partition - CPU - numba: 1.0550997257232666
# 4.769582996848387e+74
#  Partition - CPU - numpy:          0.21292614936828613
# 4.769582996848387e+74
#  Partition - CPU - numpy + numba:  0.2262575626373291
# 4.769582996848387e+74
#  Partition - GPU - loop:  0.24902629852294922
# inf
