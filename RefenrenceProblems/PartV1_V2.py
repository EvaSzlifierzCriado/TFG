import numpy as np
from math import e
import numba.cuda as cuda
from time import time
import numba
from numba import jit, cuda   ### njit, vectorization

# Input:

W: [[0.92400234]
 [0.14357284]]
b: [0.61821895 0.24811802]
c: [0.04303408]
x: [[0. 0.]
 [0. 1.]
 [1. 0.]
 [1. 1.]]
h: [[0.]
 [1.]]

###############################################

#V2
def CPU_loop_PartitionV2(W,b,c,x,h):
    Prod_h_x = 0
    c_transp = np.transpose(c)
    for j in range(2**m): # Sum h 
        Prod_x = 1
        for i in range(2**n): # Product of f_i(a_i) + f_i(b_i) N times, being N the number of digits
            sum_x = 0
            for k in range(n):
                sum_x += e**(x[i][k]*(CPU_loop_ArrayMul(W[k],h[j]) + b[k]))
                # print("i,k:",i,k)
                # print("x[i][k]:",x[i][k])
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
        print("sum_xh:", sum_xh)
    return sum_xh


print("PartitionV1:", CPU_loop_Partition(W,b,c,x,h))
print("PartitionV2:", CPU_loop_PartitionV2(W,b,c,x,h))

###############################################


# La v1 me da 21.205123038575923 y la v2 221.6946843184736
