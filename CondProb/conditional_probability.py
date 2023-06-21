from time import time
import numpy as np
from scipy.special import logsumexp
import itertools
from numba import cuda, jit
import math
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

@jit
def conditional_probability_numba(type_rbm, x):
    if type_rbm == 0:
        return 1 / (1 + np.exp(-x))
    else:
        return (1 + np.tanh(x)) / 2   ### ==  1 / (1 + np.exp(-2*x))

@cuda.jit
def conditional_probability_gpu(type_rbm, x, p):
        i = cuda.grid(1)
        if i < x.shape[0]:
          if type_rbm == 0:
            p[i] = 1 / (1 + math.exp(-x[i]))
          else:
            p[i] = (1 + math.tanh(x[i])) / 2   ### ==  1 / (1 + np.exp(-2*x))


################################ Values: #################################

# Initial value for the vars:
num_vis_array  = [15, 20, 30, 40, 60, 100, 200, 300, 500, 1000, 2000, 5000, 7000, 9000, 10000, 20000, 50000, 100000, 500000] #15

#num_runs = 5                 # number of chains
num_vis = num_vis_array[0]
b_base = np.zeros(num_vis)    # initial AIS bias values

type_rbm = 0

# CUDA vars:
threadsperblock = (16,16)
blockspergrid = (32, 32)

# Vars for execution:
R = 30
NumIter = len(num_vis_array)

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

  num_vis  = num_vis_array[iter]
  b_base = np.zeros(num_vis)   # initial AIS bias values

  # Numpy:
  tic = time()
  for i in range(R):
    Res = conditional_probability(type_rbm, +b_base)

  exec = time() - tic
  Numpy_exec.append(exec)

  # print(" Conditional Probability - CPU - numpy:         {}", exec)
  #print(np.tile(Res, [num_runs, 1]))

  # Numba + Numpy:
  tic = time()
  Res = conditional_probability_numba(type_rbm, +b_base) # Compilation
  tic2 = time()
  for i in range(R):
    Res = conditional_probability_numba(type_rbm, +b_base)

  exec = time() - tic2
  comp = time() - tic

  Numba_exec.append(exec)
  Numba_comp.append(comp)

  # print(" Conditional Probability - CPU - numpy + numba (exec): {}", exec)
  # print(" Conditional Probability - CPU - numpy + numba: {}", comp)
  #print(np.tile(Res, [num_runs, 1]))

  # CUDA:
  b_base_cuda = cuda.to_device(+b_base)
  p_cuda = cuda.to_device(np.zeros(b_base.size))

  tic = time()
  conditional_probability_gpu[blockspergrid, threadsperblock](type_rbm, b_base_cuda, p_cuda) # Compilation
  tic2 = time()
  for i in range(R):
    conditional_probability_gpu[blockspergrid, threadsperblock](type_rbm, b_base_cuda, p_cuda)

  exec = time() - tic2
  comp = time() - tic

  Cuda_exec.append(exec)
  Cuda_comp.append(comp)

  # print(" Conditional Probability - GPU - CUDA (exec): {}", exec)
  # print(" Conditional Probability - GPU - CUDA: {}", comp)
#print(np.tile(p_cuda, [num_runs, 1]))

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
plt.title("Tiempos de compilación y ejecución - Conditional Probability", pad=20)
plt.yscale('log')
plt.ylabel("Tiempo en ms", labelpad=20)
plt.xlabel("Valor de N", labelpad=10)
ax.set_xticks(range(NumIter))
ax.set_xticklabels(num_vis_array, rotation=45, ha='right')

# plt.xticks(num_vis_array, rotation=45, ha='right')
#plt.yticks(y)
ax.legend()
plt.show()

