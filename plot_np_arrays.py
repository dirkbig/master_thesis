from blockchain.smartcontract import *
from source.function_file import *
import sys
import numpy as np
from mesa import Agent, Model
from pyswarm import pso
import numpy.ma as ma

#
#
# c_n_iterations = np.load('/Users/dirkvandenbiggelaar/Desktop/python_plots/np_arrays/c_nominal_list_41.npy', mmap_mode='r')
# w_n_iterations = np.load('/Users/dirkvandenbiggelaar/Desktop/python_plots/np_arrays/w_nominal_list_41.npy', mmap_mode='r')
# plt.plot(c_n_iterations)
# plt.plot(w_n_iterations)
# plt.show()


c_n_avg_iterations = np.load('/Users/dirkvandenbiggelaar/Desktop/python_plots/np_arrays/c_nominal_avg_list64.npy', mmap_mode='r')
w_n_avg_iterations = np.load('/Users/dirkvandenbiggelaar/Desktop/python_plots/np_arrays/w_nominal_avg_list64.npy', mmap_mode='r')

plt.plot(c_n_avg_iterations)
plt.plot(w_n_avg_iterations)
plt.show()