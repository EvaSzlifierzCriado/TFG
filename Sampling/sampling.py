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

def sampling(type_rbm, p, rand):
        """Sample an array of floating point elements
        
        Parameters
        ----------
            p : n dimensional numpy.array
                contains floating point numbers between 0 and 1

        Returns
        -------
            s : n dimensional numpy.array
                sampled values that are 1 with probability p and {0, -1} 
                otherwise
        """
        # Check if probability to be up is larger than a random number
        #rand = np.random.rand(*p.shape)
        idx_up = p - rand > 0
        # Put up neurons to 1 and the rest to 0 or -1
        if type_rbm == 0:
            s = np.zeros(p.shape)
            s[idx_up] = 1
        else:
            s = -np.ones(p.shape)
            s[idx_up] = 1
        return s

@jit
def sampling_numba(type_rbm, p, rand):
    # Check if probability to be up is larger than a random number
    s = np.where(p - rand > 0, 1, 0)
    if type_rbm != 0:
        s = s - 1
    return s

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

# Initial value for the vars:
num_vis_array  = [15, 20, 30, 40, 60, 100, 200, 300, 500, 1000, 2000, 5000, 7000, 9000, 10000] #15
num_runs_array = [5, 7, 10, 15, 20, 30, 40, 50, 60, 80, 100, 200, 300, 400, 500]

#num_runs = 5                 # number of chains # 5

num_vis = num_vis_array[0]
num_runs = num_runs_array[0]
b_base = np.zeros(num_vis)  # initial AIS bias values

type_rbm = 0
pvdata = np.tile(conditional_probability(type_rbm, +b_base), [num_runs, 1]) 

# Additional values:
random_numbers_matrix = np.random.rand(*pvdata.shape)

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
  num_runs = num_runs_array[iter]                 # number of chains
  b_base = np.zeros(num_vis)   # initial AIS bias values
  pvdata = np.tile(conditional_probability(type_rbm, +b_base), [num_runs, 1]) 
  # Additional values:
  random_numbers_matrix = np.random.rand(*pvdata.shape) 


  # Numpy:
  tic = time()
  for i in range(R):
    cpu_res = sampling(type_rbm, pvdata, random_numbers_matrix)
  # print(cpu_res)

  exec = time() - tic
  Numpy_exec.append(exec)


  # Numba + Numpy:
  tic = time()
  cpu_numba_res = sampling_numba(type_rbm, pvdata, random_numbers_matrix) # Compilation
  tic2 = time()
  for i in range(R):
    cpu_numba_res = sampling_numba(type_rbm, pvdata, random_numbers_matrix)
  
  exec = time() - tic2
  comp = time() - tic

  Numba_exec.append(exec)
  Numba_comp.append(comp)


  # CUDA:
  cuda_pvdata = cuda.to_device(pvdata)
  cuda_random_numbers_matrix = cuda.to_device(random_numbers_matrix)
  s_cuda = cp.zeros(pvdata.shape)

  tic = time()
  sampling_gpu[blockspergrid, threadsperblock](type_rbm, cuda_pvdata, cuda_random_numbers_matrix, s_cuda)
  tic2 = time()
  for i in range(R):
    sampling_gpu[blockspergrid, threadsperblock](type_rbm, cuda_pvdata, cuda_random_numbers_matrix, s_cuda)
  #print(s_cuda)

  exec = time() - tic2
  comp = time() - tic

  Cuda_exec.append(exec)
  Cuda_comp.append(comp)

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
plt.title("Tiempos de compilación y ejecución - Sampling", pad=20)
plt.yscale('log')
plt.ylabel("Tiempo en ms", labelpad=20)
plt.xlabel("Valor de NxM", labelpad=10)
ax.set_xticks(range(NumIter))

# Creation of the labels for axis x:
labels = []
for k in range(NumIter):
  i = num_vis_array[k]
  j = num_runs_array[k]
  labels.append(str(i) + 'x' + str(j))

ax.set_xticklabels(labels, rotation=45, ha='right')

# plt.xticks(num_vis_array, rotation=45, ha='right')
#plt.yticks(y)
ax.legend()
plt.show()


