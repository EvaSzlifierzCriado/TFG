from time import time
import numpy as np
from scipy.special import logsumexp
import itertools
from time import time
from numba import jit, float64, int64, types, njit
from numba import cuda
import math
import matplotlib.pyplot as plt

def log_f_expert(type_rbm, x):
        idx_big = np.abs(x) > 20
        idx_low = ~idx_big
        y = np.empty(x.shape)
        if type_rbm == 0:
            # Rectifier unit
            y[idx_big] = x[idx_big]*(x[idx_big]>0)
            y[idx_low] = np.log(1 + np.exp(x[idx_low]))
        else:
            # Absolute value unit
            y[idx_big] = np.abs(x[idx_big])
            y[idx_low] = np.log(2*np.cosh(x[idx_low]))
        return y

@jit(nopython=True)
def log_f_expert_numba(type_rbm, x):
    y = np.empty_like(x)
    for i in range(x.shape[0]):
        if np.abs(x[i]) > 20:
            if type_rbm == 0:
                # Rectifier unit
                y[i] = x[i] * (x[i] > 0)
            else:
                # Absolute value unit
                y[i] = np.abs(x[i])
        else:
            if type_rbm == 0:
                # Log-exp unit
                y[i] = np.log(1 + np.exp(x[i]))
            else:
                # Log-cosh unit
                y[i] = np.log(2 * np.cosh(x[i]))
    return y

@cuda.jit
def log_f_expert_cuda(type_rbm, x, y):
        i = cuda.grid(1)
        if i < x.shape[0]:
          if abs(x[i] > 10):
              if type_rbm == 0:
                  # Rectifier unit
                  if x[i] > 0:
                      y[i] = x[i]
                  else:
                      y[i] = 0
              else:
                    # Absolute value unit
                    y[i] = abs(x[i])
          else:
              if type_rbm == 0:
                  # Log-exp unit
                  y[i] = math.log(1 + math.exp(x[i]))
              else:
                  # Log-cosh unit
                  y[i] = math.log(2 * math.cosh(x[i]))


################################ Values: #################################

# Initial value for the vars:
num_vis_array  = [15, 20, 30, 40, 60, 100, 200, 300, 500, 1000, 2000, 5000, 7000, 9000, 10000, 20000, 50000, 100000, 250000, 500000, 600000, 700000, 800000, 900000, 1000000] #15
num_vis = num_vis_array[0]
b_base = np.zeros(num_vis)    # initial AIS bias values
type_rbm = 0

# CUDA vars:
threadsperblock = (20, 20)
blockspergrid = (25000)

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
    res = log_f_expert(type_rbm, b_base)
  
  exec = time() - tic
  Numpy_exec.append(exec)


  # Numba:
  tic = time()
  res = log_f_expert_numba(type_rbm, b_base)
  tic2 = time()
  for i in range(R):
    res = log_f_expert_numba(type_rbm, b_base)

  exec = time() - tic2
  comp = time() - tic

  Numba_exec.append(exec)
  Numba_comp.append(comp)


  # Cuda Vars:
  y_cuda = np.empty(b_base.shape)
  b_base_cuda = cuda.to_device(b_base)

  tic = time()
  log_f_expert_cuda[blockspergrid, threadsperblock](type_rbm, b_base_cuda, y_cuda)
  tic2 = time()
  for i in range(R):
    log_f_expert_cuda[blockspergrid, threadsperblock](type_rbm, b_base_cuda, y_cuda)
  
  exec = time() - tic2
  comp = time() - tic

  Cuda_exec.append(exec)
  Cuda_comp.append(comp)
  #print(y_cuda)


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
plt.title("Tiempos de compilación y ejecución - Log F Expert", pad=20)
plt.yscale('log')
plt.ylabel("Tiempo en ms", labelpad=20)
plt.xlabel("Valor de N", labelpad=10)
ax.set_xticks(range(NumIter))
ax.set_xticklabels(num_vis_array, rotation=45, ha='right')

# plt.xticks(num_vis_array, rotation=45, ha='right')
#plt.yticks(y)
ax.legend()
plt.show()

