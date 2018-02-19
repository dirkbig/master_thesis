import numpy as np
from pyswarm import pso
from source.function_file import *
import random

N = 3

""" Parameters Buyers"""
E_i = [30, 50, 30]
lambda11 = [2, 2, 2]
lambda12 = [0.7, 0.7, 0.7]
E_j = 1000 # 100 units to be dedicated this step


args_buyer = [E_i, E_j, lambda11, lambda12]

""" Parameters Sellers"""
def buyer_objective_function_PSO(c, *args_buyer):
    """ Costfunction Buyer PSO"""
    cost_buyers = 0
    """ agents specific weights"""
    E_i = args_buyer[0]
    E_j = args_buyer[1]
    lambda11_arg= args_buyer[2]
    lambda12_arg = args_buyer[3]
    c_l_opt = sum(c)

    for i in range(N):
        E_i_opt = E_i[i] # demand per agent (buying)
        _lambda11 = lambda11_arg[i]
        _lambda12 = lambda12_arg[i]

        E_j_opt = E_j # supply of all sellers (external)

        cost_buyers += abs(E_i_opt - E_j_opt * c[i] / (c_l_opt + c[i])) ** _lambda11 + (c[i] * E_j_opt * (c[i] / (c_l_opt + c[i]))) ** _lambda12

    return cost_buyers
def seller_objective_function_PSO(w, *args_seller):
    """ Costfunction Sellers PSO"""
    cost_sellers = 0

    """ agents specific weights"""

    R_direct = args_seller[0]           # predicted future profit
    E_surplus_list = args_seller[1]     # agents surplus energy
    R_future_list = args_seller[2]      # deterministic direct profit
    E_predicted_list = args_seller[3]        # predicted future surplus energy

    lambda21 = args_seller[4]
    lambda22 = args_seller[5]

    for j in range(N):
        E_surplus_j = E_surplus_list[j]
        R_future_j = R_future_list[j]
        E_predicted_j = E_predicted_list[j]

        _lambda21 = lambda21[j]
        _lambda22 = lambda22[j]

        cost_sellers += - ( (R_future_j * (E_surplus_j * (1 - w[j]) / (E_predicted_j + E_surplus_j * (1 - w[j])))) ** _lambda21 + \
                            (R_direct * (E_surplus_j * w[j] / (np.dot(w, E_surplus_list)))) ** _lambda22 )

    return cost_sellers


max_c_i = 1000
lb = np.zeros(N)
ub = np.zeros(N)
lb_seller = np.zeros(N)
ub_seller = np.zeros(N)

for i in range(N):
    lb[i] = 0
    ub[i] = max_c_i

for j in range(N):
    lb_seller[j] = 0
    ub_seller[j] = 1

while True:
    """ buyers level game """
    E_demand_list = [30, 50, 30]
    E_supply = 1000                      #    100 units to be dedicated this step
    lambda11 = [2, 2, 2]
    lambda12 = [0.7, 0.7, 0.7]

    args_buyer = [E_demand_list, E_supply, lambda11, lambda12]
    c_opt, f_opt = pso(buyer_objective_function_PSO, lb, ub, ieqcons=[], args=args_buyer, swarmsize=1000, omega=0.9, phip=0.6,
                   phig=0.6, maxiter=1000, minstep=1e-1, minfunc=1e-1)  #
    result_buyers = c_opt
    print(c_opt)

    """ sellers level game """
    R_direct = np.dot(E_demand_list, c_opt)
    E_surplus_list = [20, 40, 60]
    R_future_list = [25, 50, 75]
    E_predicted_list = [100, 120, 130]
    lambda21 = [2, 2, 2]
    lambda22 = [2, 2, 2]
    args_seller = [R_direct, E_surplus_list, R_future_list, E_predicted_list, lambda21, lambda22]

    w_opt, f_opt = pso(seller_objective_function_PSO, lb_seller, ub_seller, ieqcons=[], args=args_seller, swarmsize=1000, omega=0.9, phip=0.6,
                   phig=0.6, maxiter=1000, minstep=1e-1, minfunc=1e-1)  #
    result_sellers = w_opt
    print(w_opt)

    result_round = [c_opt, w_opt]







""" cost function """