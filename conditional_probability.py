import numpy as np
from scipy.special import logsumexp
import itertools
from numba import cuda
import math

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
def conditional_probability_gpu(type_rbm, x, p):
        i = cuda.grid(1)
        if i < x.shape[0]:
          if type_rbm == 0:
            p[i] = 1 / (1 + math.exp(-x[i]))
          else:
            p[i] = (1 + math.tanh(x[i])) / 2   ### ==  1 / (1 + np.exp(-2*x))


num_vis  = 15

num_runs = 5                 # number of chains
b_base = np.zeros(num_vis)  # initial AIS bias values

type_rbm = 0

print(np.tile(conditional_probability(type_rbm, +b_base), [num_runs, 1]))

# CUDA vars:
threadsperblock = (15,15)
blockspergrid = (1,1)

b_base_cuda = cuda.to_device(+b_base)
p_cuda = cuda.to_device(np.zeros(b_base.size))

conditional_probability_gpu[blockspergrid, threadsperblock](type_rbm, b_base_cuda, p_cuda)
print(np.tile(p_cuda, [num_runs, 1]))

