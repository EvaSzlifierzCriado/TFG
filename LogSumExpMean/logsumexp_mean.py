from time import time
import numpy as np
from scipy.special import logsumexp
import itertools
from numba import cuda, jit
import math
import cupy as cp
import matplotlib.pyplot as plt

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

@jit(nopython=True)
def logsumexp_mean_numba(v):
    n = len(v)
    r = np.log(np.sum(np.exp(v))) - np.log(n)
    return r

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

################################ Values: #################################

# Initial value for the vars:
sizes  = [15, 20, 30, 40, 60, 100, 200, 300, 500, 1000, 2000, 5000, 7000, 9000, 10000, 20000, 50000, 100000, 500000] #15

# Vars:
#   Example of log_w array:
log_w = np.random.rand(sizes[0])

# CUDA vars:
threadsperblock = (16,16)
blockspergrid = (32, 32)

# Vars for execution:
R = 30
NumIter = len(sizes)

# The result times (ex AND ex+comp) for each function will be stored in these arrays:
Numpy_exec = []

Numba_comp = []
Numba_exec = []

Cuda_comp = []
Cuda_exec = []

################################ Execution: #################################

# NumIter is the number of times the functions will be executed.
# Each iteration will add a value for the plot.
for iter in range(NumIter):

  log_w = np.random.rand(sizes[iter])

  # Numpy
  tic = time()
  for i in range(R):
    res = logsumexp_mean(log_w)
  
  exec = time() - tic
  Numpy_exec.append(exec)


  # Numba:
  tic = time()
  res = logsumexp_mean_numba(log_w) # Compilation
  tic2 = time()
  for i in range(R):
    res = logsumexp_mean_numba(log_w)
  
  exec = time() - tic2
  comp = time() - tic

  Numba_exec.append(exec)
  Numba_comp.append(comp)


  # CUDA:
  log_w_cuda = cuda.to_device(log_w)
  temp = np.empty((log_w.shape), dtype=np.float64)
  temp_cuda = cuda.to_device(temp)
  r_cuda = cuda.to_device(np.empty(1, dtype=np.float64))

  tic = time()
  logsumexp_mean_cuda[blockspergrid, threadsperblock](log_w_cuda, temp_cuda, r_cuda) # Compilation
  tic2 = time()
  for i in range(R):
    logsumexp_mean_cuda[blockspergrid, threadsperblock](log_w_cuda, temp_cuda, r_cuda)
    #temp2 = temp_cuda.copy_to_host()
  
  exec = time() - tic2
  comp = time() - tic

  Cuda_exec.append(exec)
  Cuda_comp.append(comp)


  #r = r_cuda.copy_to_host()
# print(r[0])

################################ Plot printing: #################################

# Convert the results to miliseconds:
for i in Numpy_exec:
  i = i*1000
for i in Numba_exec:
  i = i*1000
for i in Numba_comp:
  i = i*1000
for i in Cuda_exec:
  i = i*1000
for i in Cuda_comp:
  i = i*1000


fig, ax = plt.subplots(figsize=(12, 6))

# We need to draw the canvas, otherwise the labels won't be positioned and 
# won't have values yet.
fig.canvas.draw()

#fig.figure(figsize=(10, 6))


ax.plot(range(NumIter), Numpy_exec, c="orange", label = "Numpy")

ax.plot(range(NumIter), Numba_comp, c="blue", label = "Numba compilación + ejecución")
ax.plot(range(NumIter), Numba_exec, c="cyan", label = "Numba ejecución")

ax.plot(range(NumIter), Cuda_comp, c="olive", label = "CUDA compilación + ejecución")
ax.plot(range(NumIter), Cuda_exec, c="green", label = "CUDA ejecución")


#plt.xticks(x_exec, rotation=45, ha='right')
plt.title("Tiempos de compilación y ejecución - Log Sum Exp Mean", pad=20)
plt.yscale('log')
plt.ylabel("Tiempo en ms", labelpad=20)
plt.xlabel("Valor de N", labelpad=10)
ax.set_xticks(range(NumIter))
ax.set_xticklabels(sizes, rotation=45, ha='right')

# plt.xticks(num_vis_array, rotation=45, ha='right')
#plt.yticks(y)
ax.legend()
plt.show()
