import csv
import numpy as np
from scipy.optimize import minimize

# Init
"""Initialization of parameters
demand_i = 0                                # E_i, total demand of player i at current time_step
c_S = 5                                     # c_S is selling price of the microgrid
c_B = 10                                    # c_B is buying price of the microgrid
C_i = [c_S, c_B]                            # Domain of available prices for bidding player i
c_i = 10                                    # Bidding price per player c_i
possible_c_i = range(C_i[0], C_i[1])        # domain of solution for bidding price c_i

c_i_price_vector = 0
w_j_storage_factor = 0
total_energy_surplus = 0

"""


def read_csv(filename,duration):                                    # still using a single load pattern for all houses
    """Reads in load and generation data from dataset"""
    with open(filename) as csvfile:
        CSVread = csv.reader(csvfile, delimiter=',')
        data_return = np.array([])
        for row in CSVread:
            data_value = float(row[-1])
            data_return = np.append(data_return, data_value)
        while len(data_return) < duration:
            data_return = np.append(data_return, 0)
        return data_return


def define_pool(consumption_at_round, production_at_round):
    """this function has to decide whether agent is a buyer or a seller"""
    surplus_per_agent = production_at_round - consumption_at_round
    if surplus_per_agent > 0:
        classification = "seller"
        demand_agent = 0
    elif surplus_per_agent < 0:
        classification = "buyer"
        demand_agent = - surplus_per_agent
        surplus_per_agent = 0
    else:
        classification = "passive"
    return [classification, surplus_per_agent, demand_agent]


def calc_supply(surplus_per_agent, w_j_storage_factor):             # E_j
    supply_per_agent = surplus_per_agent * w_j_storage_factor
    return supply_per_agent


def calc_demand(demand_per_agent, ):                                # E_i

    pass


def allocation_to_i_func(supply_on_step, bidding_price_i, bidding_prices_all):            # w_j_storage_factor is nested in supply_on_step
    E_i = supply_on_step*(bidding_price_i/(sum(bidding_prices_all)))
    return E_i


def calc_revenue_j(payment_i, demand_i, c_i, total_energy_surplus):
    """calculation of revenue for seller j depending on allocation to i"""
    # for agent in
    revenue_total = sum(payment_i,[0])
    revenue_j = revenue_total * (demand_i*c_i)/total_energy_surplus    # payment_i is E_i*ci = amount of demanded energy * bidding price
    return revenue_j


def calc_utility_function_i(bidding_price, demand_i):
    """This function calculates buyers utility"""
    utility_i = demand_i * (bidding_price[0] - bidding_price)
    return utility_i


def calc_utility_function_j(estimated_energy_j, w_j_storage_factor, revenue_j):
    """This function calculates sellers utility"""
    e_j_energy_surplus_j = estimated_energy_j * w_j_storage_factor         # \hat(E)_j surplus energy of selling agent j
    utility_j = e_j_energy_surplus_j*(1 - w_j_storage_factor) + revenue_j  # utility of selling agent j
    return utility_j


def buyers_game_optimization(supply_on_step, bidding_price_i_prev, bidding_prices_all, C_i):
    """Level 1 game: distributed optimizaton"""
    E = supply_on_step
    c_S = C_i[0]
    c_l_total = bidding_prices_all

    def utility_buyer(E, c_i, c_S, c_l_total, sign= -1):
        """parametric utility function"""
        return sign*E*(c_i/c_l_total)*c_S - E*(c_i/c_l_total)*c_i

    # def constraint1():
    #     """constraints on optimization"""
    #     return []
    #
    # con1 = {'type': 'eq', 'fun': constraint1}
    # cons = [con1]
    bounds = C_i
    initial_conditions = [bidding_price_i_prev]        # previous values for
    sol = minimize(lambda c_i: utility_buyer(E, c_i, c_S, c_l_total), initial_conditions, method='SLSQP', bounds=bounds) #, constraints=cons)

    return sol.x


def sellers_game_optimization(wj, w, c):
    pass





"""testing c_i within domain C_i
if c_i_price_vector in possible_c_i:
    print("all fine")
else:
    print("macro-grid is competing")

"""






