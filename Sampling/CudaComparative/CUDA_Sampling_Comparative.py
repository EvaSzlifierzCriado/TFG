from time import time
import numpy as np
import cupy as cp
import numba
from numba import cuda, jit
import matplotlib.pyplot as plt


def conditional_probability(type_rbm, x):
        """Function to compute conditional probabilities
        It can be shown that
            p( h | v ) = conditional_probability( w * v + c )
            p( v | h ) = conditional_probability( h * w + b )
        """
        if type_rbm == 0:
            return 1 / (1 + np.exp(-x))
        else:
            return (1 + np.tanh(x)) / 2   ### ==  1 / (1 + np.exp(-2*x))



@cuda.jit
def sampling_gpu(type_rbm, p, rand, s):
    i, j = cuda.grid(2)
    if i < p.shape[0] and j < p.shape[1]:
      if type_rbm == 0:
        if p[i][j] - rand[i][j] > 0:
          s[i][j] = 1
        else:
          s[i][j] = 0
      else:
        if p[i][j] - rand[i][j] > 0:
          s[i][j] = 1
        else:
          s[i][j] = -1



################################ Values: #################################

# Vars:
num_vis =100
num_runs = 50

b_base = np.zeros(num_vis) 
type_rbm = 0
pvdata = np.tile(conditional_probability(type_rbm, +b_base), [num_runs, 1]) 

# Additional values:
random_numbers_matrix = np.random.rand(*pvdata.shape)

# CUDA vars:
cuda_pvdata = cuda.to_device(pvdata)
cuda_random_numbers_matrix = cuda.to_device(random_numbers_matrix)
s_cuda = cp.zeros(pvdata.shape)

threadsperblock_array = []

divisores = []

for num in range(256, 513):
  if num_vis * num_runs % num == 0:
    divisores.append(num)

x = 1
y = 1

for div in divisores:
  for x in range(div):
    for y in range(div):
      if x * y == div:
        threadsperblock_array.append([x,y])

blockspergrid_array = []

for i in range(len(threadsperblock_array)):
  blockspergrid_array.append(num_vis*num_runs / threadsperblock_array[i][0] * threadsperblock_array[i][1] )

print(blockspergrid_array)
print(threadsperblock_array)

print(len(blockspergrid_array))
print(len(threadsperblock_array))

NumIter = len(threadsperblock_array)

# Vars for execution:
R = 30

# The result times (ex AND ex+comp) for each iteration will be stored in these arrays:
Cuda_comp = []
Cuda_exec = []

################################ Execution: #################################

# NumIter is the number of times the functions will be executed.
# Each iteration will add a value for the plot.
for iter in range(NumIter):

  blockspergrid = int(blockspergrid_array[iter])
  threadsperblock = int(threadsperblock_array[iter][0] * threadsperblock_array[iter][1])
  
  tic = time()
  sampling_gpu[blockspergrid, threadsperblock](type_rbm, cuda_pvdata, cuda_random_numbers_matrix, s_cuda)
  tic2 = time()
  for i in range(R):
    sampling_gpu[blockspergrid, threadsperblock](type_rbm, cuda_pvdata, cuda_random_numbers_matrix, s_cuda)

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
plt.title("CUDA - Comparación de valores para tamaños de bloque y grid - Sampling", pad=20)
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