# CUDA and Numba optimization for Python

## Prerequisites

To execute this code you should install CUDA toolkit and Numba compiler as follows:

* First check the prerequisites for CUDA: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions.

* Install CUDA: https://developer.nvidia.com/cuda-downloads.

* Numba compiler: https://numba.pydata.org/numba-doc/latest/user/installing.html. 

* Install Python: https://www.python.org/downloads/.

* Reboot the system.

## Code Structure 

This code is structured as follows:

* `MatMult.py` is an implementation of different functions of the matrix multiplication problem.
* `MatConvol.py` is an implementation of different functions of the matrix convolution problem.
* `Partition.py` is an implementation of different functions of the partition problem.
* `PartitionV2.py` is another implementation of different functions of the partition problem.

Each file described above implements a loop basic function and NumPy + Numba + CUDA diverse functions for diverse Numba paramethers.

## Execution

To execute each of the files:
```bash
    python3 <file_name>.py
```

