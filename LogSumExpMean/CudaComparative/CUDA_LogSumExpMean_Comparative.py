from time import time
import numpy as np
from scipy.special import logsumexp
import itertools
from numba import cuda, jit
import math
import cupy as cp
import matplotlib.pyplot as plt

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

# Vars:
size = 50000

# CUDA vars:
threadsperblock_array = []

divisores = []

for num in range(256, 513):
    if size % num == 0:
        divisores.append(num)

x = 1
y = 1

for div in divisores:
    for x_iter in range(div):
        for y_iter in range(div):
            if x_iter * y_iter == div:
                threadsperblock_array.append([x_iter, y_iter])

blockspergrid_array = []

for i in range(len(threadsperblock_array)):
    blockspergrid_array.append(size / (threadsperblock_array[i][0] * threadsperblock_array[i][1]))

print(len(threadsperblock_array))
print(len(blockspergrid_array))

NumIter = len(threadsperblock_array)

# Vars for execution:
R = 30

# The result times (ex AND ex+comp) for each iteration will be stored in these arrays:
Cuda_comp = []
Cuda_exec = []

################################ Execution: #################################

# NumIter is the number of times the functions will be executed.
# Each iteration will add a value for the plot.
for iteration in range(NumIter):

    log_w = np.random.rand(size)
    log_w_cuda = cuda.to_device(log_w)
    temp = np.empty((log_w.shape), dtype=np.float64)
    temp_cuda = cuda.to_device(temp)
    r_cuda = cuda.to_device(np.empty(1, dtype=np.float64))

    blockspergrid = int(blockspergrid_array[iteration])
    threadsperblock = int(threadsperblock_array[iteration][0] * threadsperblock_array[iteration][1])
    tic = time()
    logsumexp_mean_cuda[blockspergrid, threadsperblock](log_w_cuda, temp_cuda, r_cuda) # Compilation
    tic2 = time()
    for i in range(R):
      logsumexp_mean_cuda[blockspergrid, threadsperblock](log_w_cuda, temp_cuda, r_cuda)
    
    exec = time() - tic2
    comp = time() - tic

    Cuda_exec.append(exec/R)
    Cuda_comp.append(comp/R)

################################ Plot printing: #################################

# Convert the results to miliseconds:
for i in Cuda_exec:
  i = i*1000
for i in Cuda_comp:
  i = i*1000


fig, ax = plt.subplots(figsize=(12, 6))

# We need to draw the canvas, otherwise the labels won't be positioned and 
# won't have values yet.
fig.canvas.draw()

#fig.figure(figsize=(10, 6))

ax.plot(range(NumIter), Cuda_comp, c="olive", label = "CUDA compilación + ejecución")
ax.plot(range(NumIter), Cuda_exec, c="green", label = "CUDA ejecución")


#plt.xticks(x_exec, rotation=45, ha='right')
plt.title("CUDA - Comparación de valores para tamaños de bloque y grid - Log Sum Exp Mean", pad=20)
plt.yscale('log')
plt.ylabel("Tiempo en ms", labelpad=20)
plt.xlabel("Tamaño de grid, tamaño de bloque", labelpad=10)
ax.set_xticks(range(NumIter))

params = []
for i in range(NumIter):
  param = str(blockspergrid_array[i]) + " , "  + str(threadsperblock_array[i][0]) + "x" + str(threadsperblock_array[i][0])
  params.append(param)

ax.set_xticklabels(params, rotation=45, ha='right')

# plt.xticks(num_vis_array, rotation=45, ha='right')
#plt.yticks(y)
ax.legend()
plt.show()

