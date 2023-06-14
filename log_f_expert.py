import numpy as np
from scipy.special import logsumexp
import itertools
from time import time
from numba import jit, float64, int64, types, njit
from numba import cuda
import math

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

# Vars:
num_vis = 15
b_base = np.zeros(num_vis)
type_rbm = 0

print(log_f_expert(type_rbm, b_base))

# Cuda Vars:
y_cuda = np.empty(b_base.shape)
b_base_cuda = cuda.to_device(b_base)

threadsperblock = (15,15)
blockspergrid = (1,1)

log_f_expert_cuda[blockspergrid, threadsperblock](type_rbm, b_base_cuda, y_cuda)
print(y_cuda)