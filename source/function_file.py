import csv
import numpy as np
from scipy.optimize import minimize
import random


def read_csv(filename,duration):                                    # still using a single load pattern for all houses
    """Reads in load and generation data from data set"""
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


def allocation_to_i_func(supply_on_step, bidding_price_i, bidding_prices_total):            # w_j_storage_factor is nested in E_total_supply
    Ei = supply_on_step*(bidding_price_i/bidding_prices_total)
    return Ei


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


def buyers_game_optimization(id_buyer, supply_on_step, c_macro, bidding_prices_all, bidding_price_i_prev):
    """Level 1 game: distributed optimization"""

    """globally declared variables, do not use somewhere else!!"""
    global E_global_buyers, c_S_global_buyers, c_l_global_buyers, c_i_global_buyers
    E_global_buyers = supply_on_step
    c_S_global_buyers = c_macro[0]
    c_l_global_buyers = bidding_prices_all
    c_i_global_buyers = bidding_price_i_prev

    initial_conditions = [E_global_buyers, c_S_global_buyers, c_l_global_buyers, c_i_global_buyers]

    def utility_buyer(x, sign=-1):
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]
        x3 = x[3]

        """parametric utility function"""
        return sign*x0*(x3/x2)*x1 - x0*(x3/x2)*x3

    """fix parameters E_global, c_S_global, c_l_global"""
    def constraint_param_x0(x):
        return E_global_buyers - x[0]

    def constraint_param_x1(x):
        return c_S_global_buyers - x[1]

    def constraint_param_x2(x):
        return c_l_global_buyers - x[2]

    """incorporate various constraints"""
    con0 = {'type': 'eq', 'fun': constraint_param_x0}
    con1 = {'type': 'eq', 'fun': constraint_param_x1}
    con2 = {'type': 'eq', 'fun': constraint_param_x2}
    cons = [con0, con1, con2]
    bounds_buyer = ((0, None), (0, None), (0, None), (c_macro[0], c_macro[1]))

    """optimize using SLSQP(?)"""
    sol_buyer = minimize(utility_buyer, initial_conditions, method='SLSQP', bounds=bounds_buyer, constraints=cons)
    # print("optimization result is a bidding price of %f" % sol.x[3])
    print("buyer %d game results in %s" % (id_buyer, sol_buyer.x[3]))

    """return 4th element of solution vector."""
    return sol_buyer.x[3]


def calc_gamma():
    return random.uniform(0, 1)
    """This function will ultimately predict storage weight
    This will involve some model predictive AI"""


def sellers_game_optimization(id_seller, total_offering, supply_energy_j, total_supply_energy, gamma, w_j_storage_factor):
    """ Anticipation on buyers is plugged in here"""

    global Ej_global_sellers, R_total_global_sellers, gamma_global_sellers, E_global_sellers, wj_global_sellers
    Ej_global_sellers = supply_energy_j
    R_total_global_sellers = total_offering
    gamma_global_sellers = gamma
    E_global_sellers = total_supply_energy
    wj_global_sellers = w_j_storage_factor

    initial_conditions_seller = [Ej_global_sellers, R_total_global_sellers, gamma_global_sellers, E_global_sellers, wj_global_sellers]

    def utility_seller(x, sign= -1):
        x0 = x[0]  # Ej_global_sellers
        x1 = x[1]  # R_total_global_sellers
        x2 = x[2]  # gamma_global_sellers
        x3 = x[3]  # E_global_sellers
        x4 = x[4]  # wj_global_sellers

        return sign * (1 - x2)*((1.0 + x0*(1 - x4)) + x2 * x1 * (x0 * x4) / x3)

    def constraint_param_seller0(x):
        return Ej_global_sellers - x[0]

    def constraint_param_seller1(x):
        return R_total_global_sellers - x[1]

    def constraint_param_seller2(x):
        return gamma_global_sellers - x[2]

    def constraint_param_seller3(x):
        return E_global_sellers - x[3]

    """incorporate various constraints"""
    con_seller0 = {'type': 'eq', 'fun': constraint_param_seller0}
    con_seller1 = {'type': 'eq', 'fun': constraint_param_seller1}
    con_seller2 = {'type': 'eq', 'fun': constraint_param_seller2}
    con_seller3 = {'type': 'eq', 'fun': constraint_param_seller3}
    cons_seller = [con_seller0, con_seller1, con_seller2, con_seller3]
    bounds_seller = ((0, None), (0, None), (0, None), (0, None), (0, 1))

    sol_seller = minimize(utility_seller, initial_conditions_seller, method='SLSQP', bounds=bounds_seller, constraints=cons_seller)  # bounds=bounds
    print("seller %d game results in %s" % (id_seller, sol_seller.x[4]))

    """return 5th element of solution vector."""
    return sol_seller.x[4]
    pass





"""testing c_i within domain C_i
if c_i_price_vector in possible_c_i:
    print("all fine")
else:
    print("macro-grid is competing")

"""






