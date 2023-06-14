import numpy as np
import cupy as cp
import numba
from numba import cuda

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

num_runs = 5                 # number of chains
b_base = np.zeros(num_vis)  # initial AIS bias values

type_rbm = 0
pvdata = np.tile(conditional_probability(type_rbm, +b_base), [num_runs, 1])  

# Additional values:
random_numbers_matrix = np.random.rand(*pvdata.shape)

cpu_res = sampling(type_rbm, pvdata, random_numbers_matrix)
print(cpu_res)

cuda_pvdata = cuda.to_device(pvdata)
cuda_random_numbers_matrix = cuda.to_device(random_numbers_matrix)
s_cuda = cp.zeros(pvdata.shape)

sampling_gpu[(5,5),(15,15)](type_rbm, cuda_pvdata, cuda_random_numbers_matrix, s_cuda)
print(s_cuda)