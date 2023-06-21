import numpy as np
from scipy.special import logsumexp
import itertools
from time import time
from numba import jit, float64, int64, types, njit
import matplotlib.pyplot as plt

def AIS_log_Z(param):

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
        idx_up = p - rand > 0
        # Put up neurons to 1 and the rest to 0 or -1
        if type_rbm == 0:
            s = np.zeros(p.shape)
            s[idx_up] = 1
        else:
            s = -np.ones(p.shape)
            s[idx_up] = 1
        return s

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

    def logsumexp_mean (v):            # logmeanexp
        n = len(v)
        r = logsumexp(v) - np.log(n)   # r = log ( 1/N \sum exp(v_i) )
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
    pvdata = np.tile(conditional_probability(type_rbm, +b_base), [num_runs, 1])                              # ERM - Assumes a = 0 => Energy = b'v
    vdata = sampling(type_rbm, pvdata)                                                                       # ERM - negdata = v_1

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
    log_w -= (1-beta)*Energy_base + beta*vb_rbm + np.sum(log_f_expert(type_rbm, vw_c_rbm_beta), axis=1)      # ERM - adds  -log(p_0^*(v_{1}))  (beta = 0)

    # Core of AIS run
    for beta in beta_vals[1:-1]:     # secuencial                                                                        # ERM - removes first and last term (beta=0,1)
        vw_c_rbm_beta = beta * vw_c_rbm
        log_w += (1-beta)*Energy_base + beta*vb_rbm + np.sum(log_f_expert(type_rbm, vw_c_rbm_beta), axis=1)  # ERM - adds  +log(p_i^*(v_{i})) (-log(2^Nh) that cancels out)
                                                                                                             # ERM - equation (15) not needed (W_A = a = 0)
        phdata = conditional_probability(type_rbm, vw_c_rbm_beta)
        hdata = sampling(type_rbm, phdata)                                                                   # ERM - equation (16)

        pvdata = conditional_probability(type_rbm, (1-beta)*b_base + beta*(np.dot(hdata, rbm_w.T) + rbm_b))  # ERM - equation (17)
        vdata = sampling(type_rbm, pvdata)                                                                   # ERM - negdata = v_{i+1}

        vb_rbm        = np.dot(vdata, rbm_b)
        vw_c_rbm      = np.dot(vdata, rbm_w) + rbm_c
        vw_c_rbm_beta = beta * vw_c_rbm
        Energy_base   = np.dot(vdata, b_base)

        log_w -= (1-beta)*Energy_base + beta*vb_rbm + np.sum(log_f_expert(type_rbm, vw_c_rbm_beta), axis=1)  # ERM - adds  -log(p_i^*(v_{i+1})) (+log(2^Nh) that cancels out)

    # Contribution to log_w from beta[end]
    beta = beta_vals[-1]
    vw_c_rbm_beta = beta * vw_c_rbm
    log_w += (1-beta)*Energy_base + beta*vb_rbm + np.sum(log_f_expert(type_rbm, vw_c_rbm_beta), axis=1)      # ERM - adds  +log(p_k^*(v_{k}))  (beta = 1)

    log_Z_base = np.sum(log_f_expert(type_rbm, b_base)) + num_hid * np.log(2)                                # ERM - log_Z_base = log Z_A

    # Final log_Z estimation
    r = logsumexp_mean(log_w)                                                                       # ERM - r = log (1/M \sum exp (log_w_i) )
    log_Z = log_Z_base + r

    return log_Z, log_w, log_Z_base


def CPU_numpy_numba_AIS_log_Z(param):

    @jit
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


    @jit(nopython=True)
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

    @jit(nopython=True)
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



    @jit(nopython=True)
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
    pvdata = np.tile(conditional_probability(type_rbm, +b_base), [num_runs, 1]).astype('float64')                              # ERM - Assumes a = 0 => Energy = b'v
    vdata = sampling(type_rbm, pvdata)                                                                       # ERM - negdata = v_1

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
                                                                                                             # ERM - equation (15) not needed (W_A = a = 0)
        phdata = conditional_probability(type_rbm, vw_c_rbm_beta)
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
#              Logarithm of the exact partition function                  #
# ----------------------------------------------------------------------- #


def log_exact_part_fun(param):
    """Logarithm of the exact partition function
    """

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

    def unnormalized_logprob_vis(v, type_rbm, rbm_b, rbm_c, rbm_w):
        vb = np.dot(v, rbm_b)
        vw_c = np.dot(v, rbm_w) + rbm_c
        log_expert = log_f_expert(type_rbm, vw_c)
        if len(log_expert.shape) == 2:
            return vb + np.sum(log_expert, axis=1)
        else:
            return vb + np.sum(log_expert)

    def unnormalized_logprob_hid(h, type_rbm, rbm_b, rbm_c, rbm_w):
        hc = np.dot(h, rbm_c)
        hw_b = np.dot(h, rbm_w.T) + rbm_b
        log_expert = log_f_expert(type_rbm, hw_b)
        return hc + np.sum(log_expert, axis=1)


    type_rbm = param['type_RBM']
    rbm_b    = param['RBM_b']
    rbm_c    = param['RBM_c']
    rbm_w    = param['RBM_w']
    num_vis  = len(rbm_b)
    num_hid  = len(rbm_c)
    
    num_neurons = min(num_vis, num_hid)
    states = itertools.product([-type_rbm, 1], repeat=num_neurons)
    unnormalized_logprob = np.empty(2**num_neurons)
    i = 0
    for state in states:
        s = np.array(state).reshape([1, num_neurons])
        if num_vis <= num_hid:
            unnormalized_logprob[i] = unnormalized_logprob_vis(s, type_rbm, rbm_b, rbm_c, rbm_w)
        else:
            unnormalized_logprob[i] = unnormalized_logprob_hid(s, type_rbm, rbm_b, rbm_c, rbm_w)
        i += 1
        logZ = logsumexp(unnormalized_logprob)

    return logZ

# ----------------------------------------------------------------------- #
#                                 Main                                    #
# ----------------------------------------------------------------------- #

# Vectors for saving the time development when changing the values of the functions:
AIS_numpy_time = []

AIS_numpy_numba_ex_time = []

AIS_numpy_numba_comp_time = []

################################ Values: #################################

R = 30
NumIter = [0, 1]

# NumIter is the number of times the functions will be executed.
# Each iteration will add a value for the plot.
for iter in range(len(NumIter)):

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

  x_comp = ["CPU - numpy + numba"]
  x_exec = ["CPU - numpy", "CPU - numpy + numba"]
  y_comp = []
  y_exec = []

  for i in range(R):
    log_Z_exact = log_exact_part_fun(param)
  print("Log Z exact:", log_Z_exact)


  ### CPU - numpy
  tic = time()
  for i in range(1):
      log_Z_AIS, log_w, log_Z_base = AIS_log_Z(param)

  exec = time() - tic
  #y_exec.append(exec)


  print(" AIS - CPU - numpy (exec):          {}", exec)
  print(log_Z_AIS)

  AIS_numpy_time.append(exec)

  ### CPU - numpy + numba
  tic = time()
  log_Z_AIS, log_w, log_Z_base = CPU_numpy_numba_AIS_log_Z(param)
  tic2 = time()
  for i in range(1):
      log_Z_AIS, log_w, log_Z_base = CPU_numpy_numba_AIS_log_Z(param)

  exec = time() - tic2
  comp = time() - tic

  #y_exec.append(exec)
  #y_comp.append(comp)

  print(" AIS - CPU - numpy + numba (exec):          {}", exec)
  print(" AIS - CPU - numpy + numba:          {}", comp)
  print(log_Z_AIS)

  AIS_numpy_numba_ex_time.append(exec)
  AIS_numpy_numba_comp_time.append(comp)

print(AIS_numpy_time)
print(AIS_numpy_numba_ex_time)
print(AIS_numpy_numba_comp_time)

# ----------------------------------------------------------------------- #
#                        Plot printing:                                   #
# ----------------------------------------------------------------------- #


# Convert the results to miliseconds:
for i in AIS_numpy_time:
  i = i*1000

for j in AIS_numpy_numba_ex_time:
  j = j*1000

for k in AIS_numpy_numba_comp_time:
  k = k*1000


plt.figure(figsize=(10, 6))
#plt.vlines(x_exec, 0, y_exec, linestyle="dashed", colors="purple")

plt.plot(NumIter, AIS_numpy_time, c="orange", label = "Numpy", marker='o')
plt.plot(NumIter, AIS_numpy_numba_comp_time, c="blue", label="Numpy + Numba (ejecución + compilación)", marker='o')
plt.plot(NumIter, AIS_numpy_numba_ex_time, c="green", label="Numpy + Numba (ejecución)", marker='o')

#plt.xticks(x_exec, rotation=45, ha='right')
plt.title("Tiempos de compilación y ejecución - AIS", pad=20)
#plt.yscale('log')
plt.ylabel("Tiempo en ms", labelpad=20)
plt.xlabel("Iteración", labelpad=10)
plt.xticks(NumIter)
#plt.yticks(y)
plt.legend()
plt.show()


# # Convert the results to miliseconds:
# for i in y_exec:
#   i = i*1000

# plt.figure(figsize=(10, 6))
# plt.vlines(x_exec, 0, y_exec, linestyle="dashed", colors="purple")
# plt.scatter(x_exec, y_exec, c="purple")
# plt.xticks(x_exec, rotation=45, ha='right')
# plt.title("Tiempos de ejecución - AIS", pad=20)
# plt.yscale('log')
# plt.ylabel("Tiempo en ms", labelpad=20)
# plt.xlabel("Implementación", labelpad=10)
# plt.show()

# # Execution and compilation time plot:

# # Convert the results to miliseconds:
# for i in y_comp:
#   i = i*1000

# plt.figure(figsize=(10, 6))
# plt.vlines(x_comp, 0, y_comp, linestyle="dashed", colors="pink")
# plt.scatter(x_comp, y_comp, c="pink")
# plt.xticks(x_comp, rotation=45, ha='right')
# plt.title("Tiempos de compilación y ejecución - AIS", pad=20)
# plt.yscale('log')
# plt.ylabel("Tiempo en ms", labelpad=20)
# plt.xlabel("Implementación", labelpad=10)
# plt.show()

# print(len(y_exec))
# print(len(y_comp))