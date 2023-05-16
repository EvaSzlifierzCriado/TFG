import numpy as np
from math import e
import numba.cuda as cuda
from time import time
import numba
from numba import jit, cuda   ### njit, vectorization
import math

# Partition Vars:
n = 6
m = 4

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
            sum += A[j] * B[j][i]
        res.append(sum)
    return(res)

# V1:
def CPU_loop_Partition(W, b, c, x, h): 
    sum_xh = 0
    for i in range(2**n):
        for j in range(2**m):
            sum_xh += e**(CPU_loop_ArrayMul(np.transpose(b), x[i]) + CPU_loop_ArrayMul(np.transpose(c), h[j]) + CPU_loop_ArrayMul(CPU_loop_ArrayMatMul(np.transpose(x[i]), W), h[j]))
    return sum_xh

print(np.dtype(W[0][0]))
print(np.dtype(b[0]))
print(np.dtype(c[0]))
print(np.dtype(x[0][0]))
print(np.dtype(h[0][0]))
print(np.dtype(BIN[0]))


##2. CPU + Numba Partition
@jit(nopython=False, forceobj=True)
def CPU_loop_numba_PartitionV2(W,b,c,x,h):
    Prod_h_x = 0
    c_transp = np.transpose(c)
    for j in range(2**m): # Sum h 
        Prod_x = 1
        for i in range(n): 
            sum_x = 0
            for k in range(2):
                sum_x += e**(BIN[k]*(CPU_loop_ArrayMul(W[i],h[j]) + b[i]))
            Prod_x *= sum_x
        Prod_h_x += ((e**(CPU_loop_ArrayMul(c_transp, h[j]))) * Prod_x)
    return(Prod_h_x)

##3. Numpy partition
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

##4. Numpy + numba partition
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
print(" Partition2 - CPU - loop:         {}".format(time() - tic))
print(res)

### CPU - numba
res = CPU_loop_numba_PartitionV2(W,b,c,x,h)  ### Compilation
tic = time()
for i in range(R):
    res = CPU_loop_numba_PartitionV2(W,b,c,x,h)
print(" Partition2 - CPU - numba: {}".format(time() - tic))
print(res)

### CPU - numpy
tic = time()
for i in range(R):
    res = CPU_numpy_PartitionV2(W,b,c,x,h)
print(" Partition2 - CPU - numpy:          {}".format(time() - tic))
print(res)

# ### CPU - numpy + numba
# res = CPU_numpy_numba_PartitionV2(W,b,c,x,h)   ### Compilation
# tic = time()
# for i in range(R):
#     res = CPU_numpy_numba_PartitionV2(W,b,c,x,h)
# print(" Partition - CPU - numpy + numba:  {}".format(time() - tic))
# print(res)

### GPU - loop
result = np.zeros(h.shape[0], dtype=np.float64)
threads_per_block = 6
blocks_per_grid = 16
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
for i in range(R):
    GPU_loop_PartitionV2[blocks_per_grid, threads_per_block](cW,cb,cc,cx,ch,cBIN,cresult)
    res = np.sum(cresult)
print(" Partition2 - GPU - loop:  {}".format(time() - tic))
print(res)
