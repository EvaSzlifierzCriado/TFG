import numpy as np
from scipy.special import logsumexp
import itertools
from time import time
from numba import jit, float64, int64, types, njit, cuda
import cupy as cp
import math

def CUDA_AIS_log_Z(param):

    @cuda.jit
    def conditional_probability2(type_rbm, x, result):
        # Thread id:
        i = cuda.grid(1)
        if i < x.size:
            if type_rbm == cp.int64(0):
                result[i] = 1 / (1 + math.exp(-x[i]))
            else:
                result[i] = (1 + math.tanh(x[i])) / 2

    @cuda.jit
    def conditional_probability3(type_rbm, x, result):
        # Thread id:
        i, j = cuda.grid(2)
        if i < x.size:
          if j < x[0].size:
            if type_rbm == cp.int64(0):
                result[i] = 1 / (1 + math.exp(-x[i][j]))
            else:
                result[i] = (1 + math.tanh(x[i][j])) / 2
    
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

    def sampling(type_rbm, p):
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
        rand = np.random.rand(*p.shape)
        s = np.where(p - rand > 0, 1, 0)
        if type_rbm != 0:
            s = s - 1
        return s


    def log_f_expert(type_rbm, x):
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

    def log_f_expert_bidimensional(type_rbm, x):
        y = np.empty_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if np.abs(x[i, j]) > 20:
                    if type_rbm == 0:
                        # Rectifier unit
                        y[i, j] = x[i, j] * (x[i, j] > 0)
                    else:
                        # Absolute value unit
                        y[i, j] = np.abs(x[i, j])
                else:
                    if type_rbm == 0:
                        # Log-exp unit
                        y[i, j] = np.log(1 + np.exp(x[i, j]))
                    else:
                        # Log-cosh unit
                        y[i, j] = np.log(2 * np.cosh(x[i, j]))
        return y


    def logsumexp_mean(v):
        n = len(v)
        r = np.log(np.sum(np.exp(v))) - np.log(n)
        return r


    # Parameters of this run of AIS
    type_rbm = param['type_RBM']
    rbm_b    = param['RBM_b']
    rbm_c    = param['RBM_c']
    rbm_w    = param['RBM_w']
    num_runs = param['numRunsCore']
    len_beta = param['lenBeta']
    b_base   = param['AIS_b_base']
    num_hid  = len(rbm_c)
        
    # Sample from the base-rate model
    type_rbm_cuda = cp.int64(0)
    b_base_cuda = cuda.to_device(+b_base)
    # Threads = len(param['AIS_b_base]) = 15
    threadsperblock = 5
    blockspergrid = 3
    output = np.zeros(b_base.size)
    conditional_probability2[blockspergrid, threadsperblock](type_rbm_cuda, b_base_cuda, output)
    pvdata = np.tile(output, [num_runs, 1]).astype('float64')                              # ERM - Assumes a = 0 => Energy = b'v
    vdata = sampling(type_rbm, pvdata)                                                                      # ERM - negdata = v_1

    # (log) Importance weights
    log_w = np.zeros(num_runs)

    # Beta values
    beta_vals = np.linspace(0, 1, len_beta)

    # Initial values
    beta = beta_vals[0]
    vb_rbm        = np.dot(vdata, rbm_b)
    vw_c_rbm      = np.dot(vdata, rbm_w) + rbm_c
    vw_c_rbm_beta = beta * vw_c_rbm
    Energy_base   = np.dot(vdata, b_base)


    # Contribution to log_w from beta[0]
    log_w -= (1-beta)*Energy_base + beta*vb_rbm + np.sum(log_f_expert_bidimensional(type_rbm, vw_c_rbm_beta), axis=1)      # ERM - adds  -log(p_0^*(v_{1}))  (beta = 0)

    primera = True

    # Core of AIS run
    for beta in beta_vals[1:-1]:     # secuencial                                                                        # ERM - removes first and last term (beta=0,1)
        vw_c_rbm_beta = beta * vw_c_rbm
        log_w += (1-beta)*Energy_base + beta*vb_rbm + np.sum(log_f_expert_bidimensional(type_rbm, vw_c_rbm_beta), axis=1)  # ERM - adds  +log(p_i^*(v_{i})) (-log(2^Nh) that cancels out)
                              
        # type_rbm: 0
        # vw_c_... : matriz
        #funcion tocha : matriz
        type_rbm_cuda = cp.int64(0)
        vw_c_rbm_beta_cuda = cuda.to_device(vw_c_rbm_beta)    
        phdata = np.zeros(vw_c_rbm_beta.size)
        phdata = cuda.to_device(phdata)                                                                                               # ERM - equation (15) not needed (W_A = a = 0)
        conditional_probability3[5,10](type_rbm_cuda, vw_c_rbm_beta_cuda, phdata)
        #phdata = conditional_probability(type_rbm, vw_c_rbm_beta)
        print(phdata)

        if primera == True:
          print(vw_c_rbm_beta.size)
          print(vw_c_rbm_beta[0].size)
          print(vw_c_rbm_beta)
          primera = False
        hdata = sampling(type_rbm, phdata)                                                                   # ERM - equation (16)

        pvdata = conditional_probability(type_rbm, (1-beta)*b_base + beta*(np.dot(hdata, rbm_w.T) + rbm_b))  # ERM - equation (17)
        vdata = sampling(type_rbm, pvdata)                                                                   # ERM - negdata = v_{i+1}

        vb_rbm        = np.dot(vdata, rbm_b)
        vw_c_rbm      = np.dot(vdata, rbm_w) + rbm_c
        vw_c_rbm_beta = beta * vw_c_rbm
        Energy_base   = np.dot(vdata, b_base)


        log_w -= (1-beta)*Energy_base + beta*vb_rbm + np.sum(log_f_expert_bidimensional(type_rbm, vw_c_rbm_beta), axis=1)  # ERM - adds  -log(p_i^*(v_{i+1})) (+log(2^Nh) that cancels out)

    # Contribution to log_w from beta[end]
    beta = beta_vals[-1]
    vw_c_rbm_beta = beta * vw_c_rbm
    log_w += (1-beta)*Energy_base + beta*vb_rbm + np.sum(log_f_expert_bidimensional(type_rbm, vw_c_rbm_beta), axis=1)      # ERM - adds  +log(p_k^*(v_{k}))  (beta = 1)

    log_Z_base = np.sum(log_f_expert(type_rbm, b_base)) + num_hid * np.log(2)                                # ERM - log_Z_base = log Z_A

    # Final log_Z estimation
    r = logsumexp_mean(log_w)                                                                            # ERM - r = log (1/M \sum exp (log_w_i) )
    log_Z = log_Z_base + r

    return log_Z, log_w, log_Z_base

# ----------------------------------------------------------------------- #
#                                 Main                                    #
# ----------------------------------------------------------------------- #

R = 30

np.random.seed(1234)

num_vis = 15
num_hid = 10
sigma = 0.1

param = {} 
param['type_RBM'] = np.int64(0)                    # 0/1
param['RBM_b'] = sigma * np.random.randn(num_vis)
param['RBM_c'] = sigma * np.random.randn(num_hid)
param['RBM_w'] = sigma * np.random.randn(num_vis, num_hid)
#
param['numRunsCore'] = 5                 # number of chains
param['lenBeta'] = 1024                  # length of every chain
param['AIS_b_base'] = np.zeros(num_vis)  # initial AIS bias values

# for i in range(R):
#   log_Z_exact = log_exact_part_fun(param)
# print("Log Z exact:", log_Z_exact)


### CPU - numpy
tic = time()
for i in range(1):
    log_Z_AIS, log_w, log_Z_base = AIS_log_Z(param)
print(" AIS - CPU - numpy:          {}".format(time() - tic))
print(log_Z_AIS)

### CPU - numpy + numba
log_Z_AIS, log_w, log_Z_base = CPU_numpy_numba_AIS_log_Z(param)
tic = time()
for i in range(1):
    log_Z_AIS, log_w, log_Z_base = CPU_numpy_numba_AIS_log_Z(param)
print(" AIS - CPU - numpy + numba:          {}".format(time() - tic))
print(log_Z_AIS)


### GPU CUDA
log_Z_AIS, log_w, log_Z_base = CUDA_AIS_log_Z(param)
tic = time()
for i in range(1):
    log_Z_AIS, log_w, log_Z_base = CUDA_AIS_log_Z(param)
print(" AIS - GPU - CUDA:          {}".format(time() - tic))
print(log_Z_AIS)