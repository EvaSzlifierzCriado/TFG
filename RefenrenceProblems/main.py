from time import time
import numpy as np
import numba
from numba import jit, cuda   ### njit, vectorization


########################################
### Main
########################################

# Common:
R = 30

N = 100
M = 50

griddim  = 8, 8
blockdim = 16, 16

# MatMult and MatConvol vars:
A = np.random.randn(N,M)  #.astype(np.float32)
B = np.random.randn(M,N)  #.astype(np.float32)

K = 4
F = np.random.randn(K,K)  #.astype(np.float32)

# Partition Vars:
n = 10
m = 5

W = np.random.random((n,m)) # Matrix size NxM, small values
b = np.random.random((n)) # Array size N, small values
c = np.random.random((m)) # Array size M, small values

x = [] 

for i in range(0, 2**n): # Array x, size 2^n
    x_i_bin = np.binary_repr(i,n)
    x.append([int(j) for j in str(x_i_bin)])

h = []

for i in range(0, 2**m): # Array h, size 2^m
    h_i_bin = np.binary_repr(i,m)
    h.append([int(j) for j in str(h_i_bin)])

# 000
# 001
# 010
# 011
# 100
# 101
# 110
# 111 

# 8 -> 2^3 bucles
    