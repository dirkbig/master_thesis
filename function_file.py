import csv
import numpy as np

# Init
"""Initialization of parameters
demand_i = 0                           # E_i, total demand of player i at current time_step
c_S = 5                                 # c_S is selling price of the microgrid
c_B = 10                                # c_B is buying price of the microgrid
C_i = [c_S, c_B]                         # Domain of available prices for bidding player i
c_i = 10                                # Bidding price per player c_i
possible_c_i = range(C_i[0], C_i[1])     # domain of solution for bidding price c_i

c_i_price_vector = 0
w_j_storage_factor = 0
total_energy_surplus = 0

"""


def read_csv(filename,duration):
    with open(filename) as csvfile:
        CSVread = csv.reader(csvfile, delimiter=',')
        data_return = np.array([])
        for row in CSVread:
            data_value = float(row[-1])
            data_return = np.append(data_return, data_value)
        while len(data_return) < duration:
            data_return = np.append(data_return, 0)
        return data_return


def calc_revenue_j(payment_i, demand_i, c_i, total_energy_surplus):
    """calculation of revenue for seller j depending on allocation to i"""
    # for agent in
    revenue_total = sum(payment_i,[0])
    revenue_j = revenue_total * (demand_i*c_i)/total_energy_surplus    # payment_i is E_i*ci = amount of demanded energy * bidding price
    return revenue_j


def calc_utility_function_i(bidding_price, demand_i):
    """This function calculates buyers utility"""
    utility_i = demand_i * (C_i[0] - bidding_price)
    return utility_i


def calc_utility_function_j(estimated_energy_j, w_j_storage_factor, revenue_j):
    """This function calculates sellers utility"""
    e_j_energy_surplus_j = estimated_energy_j * w_j_storage_factor         # \hat(E)_j surplus energy of selling agent j
    utility_j = e_j_energy_surplus_j*(1 - w_j_storage_factor) + revenue_j  # utility of selling agent j
    return utility_j


def define_pool(consumption_at_round, production_at_round):
    """this function has to decide whether agent is a buyer or a seller"""
    supply = production_at_round - consumption_at_round
    print(supply)
    if supply > 0:
        classification = "seller"
    elif supply < 0:
        classification = "buyer"
    else:
        classification = "passive"
    return classification



"""testing c_i within domain C_i
if c_i_price_vector in possible_c_i:
    print("all fine")
else:
    print("macro-grid is competing")

"""






