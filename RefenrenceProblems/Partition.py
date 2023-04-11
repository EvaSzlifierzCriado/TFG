from main import *
from math import e

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

#1. Basic Partition Loop Function
def CPU_loop_Partition():
    Sum_x_h = 0
    c_transp = np.transpose(c)
    for i in range(2**n): # Sum x
        E_x_h = 0
        x_transp = np.transpose(x[i])
        for j in range(2**m): # Sum h
            E_x_h += CPU_loop_ArrayMul(x_transp,b) + CPU_loop_ArrayMul(c_transp,h[j]) + CPU_loop_ArrayMul(CPU_loop_ArrayMatMul(x_transp,W),h[j])
        Sum_x_h += e**E_x_h
    return(Sum_x_h)
         
print(CPU_loop_Partition())


