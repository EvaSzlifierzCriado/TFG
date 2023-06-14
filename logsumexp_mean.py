import numpy as np
from scipy.special import logsumexp
import itertools
from numba import cuda
import math
import cupy as cp

def logsumexp_mean (v):            # logmeanexp
        n = len(v)
        r = logsumexp(v) - np.log(n)   # r = log ( 1/N \sum exp(v_i) )
        return r

# @cuda.jit
# def logsumexp_mean_cuda(v, temp, sum, r):
#         i = cuda.grid(1)
#         if i < v.shape[0]:
#           temp[i] = math.exp(v[i])

#           cuda.syncthreads()
          
#           sum = sum + temp[i]

#           cuda.syncthreads()
          
#           if i == 0:
#             r[0] = math.log(sum) - math.log(v.shape[0])

@cuda.jit
def logsumexp_mean_cuda(v, temp, r):
    i = cuda.grid(1)
    n = v.shape[0]

    if i < n:
        temp[i] = math.exp(v[i])

    cuda.syncthreads()

    if i == 0:
        sum_val = 0.0
        for j in range(n):
            sum_val += temp[j]

        r[0] = math.log(sum_val) - math.log(n)

# Vars:
#   Example of log_w array:
log_w = np.array(([0.38873423, 0.37533311, 0.3920232,  0.3818863,  0.36400083]), dtype=np.float64)

print(logsumexp_mean(log_w))

# Cuda Vars:
log_w_cuda = cuda.to_device(log_w)
temp = np.empty((log_w.shape), dtype=np.float64)
temp_cuda = cuda.to_device(temp)

sum = 0.
r_cuda = cuda.to_device(np.empty(1, dtype=np.float64))

threadsperblock = (5,5)
blockspergrid = (1,1)

logsumexp_mean_cuda[blockspergrid, threadsperblock](log_w_cuda, temp_cuda, r_cuda)
temp2 = temp_cuda.copy_to_host()

r = r_cuda.copy_to_host()
print(r[0])