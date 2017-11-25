import csv
import numpy as np
from scipy.optimize import minimize
import random

import ctypes

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
        demand_agent = abs(surplus_per_agent)
        surplus_per_agent = 0
    else:
        classification = "passive"
        demand_agent = 0
        surplus_per_agent = 0
    return [classification, surplus_per_agent, demand_agent]

def bidding_prices_others(c_bidding_prices, c_i_bidding_price):
    bidding_prices_others = sum(c_bidding_prices) - c_i_bidding_price
    return bidding_prices_others


def calc_supply(surplus_per_agent, w_j_storage_factor):             # E_j
    supply_per_agent = surplus_per_agent * w_j_storage_factor
    return supply_per_agent


def calc_R_j_revenue(R_total, E_j_supply, w_j_storage_factor, E_total_supply):
    """calculation of revenue for seller j depending on allocation to i"""
    R_j = R_total * (E_j_supply*w_j_storage_factor)/E_total_supply    # payment_i is E_i*ci = amount of demanded energy * bidding price
    return R_j


def calc_utility_function_i(E_i_demand, E_total_supply, c_i_bidding_price, c_bidding_prices_others):
    """This function calculates buyers utility"""
    E_i_allocation = allocation_i(E_total_supply, c_i_bidding_price, c_bidding_prices_others)
    utility_i = (E_i_demand - (E_total_supply * (c_i_bidding_price/(c_bidding_prices_others + c_i_bidding_price))))**1.7 + (E_total_supply * (c_i_bidding_price**2/(c_bidding_prices_others + c_i_bidding_price)))
    demand_gap = E_i_demand - E_i_allocation
    return utility_i, demand_gap


def calc_utility_function_j(id, E_j_supply, R_total, E_total_supply, R_prediction, E_prediction, w_j_storage_factor):
    """This function calculates sellers utility"""
    prediction_utility = R_prediction * (E_j_supply * (1 - w_j_storage_factor) / (E_prediction + E_j_supply * (1 - w_j_storage_factor)))
    direct_utility = R_total * (E_j_supply * w_j_storage_factor / (E_total_supply + (E_j_supply * w_j_storage_factor)))
    utility_j = prediction_utility + direct_utility
    return prediction_utility, direct_utility, utility_j

    # R_j_revenue = calc_R_j_revenue(R_total, E_j_supply, w_j_storage_factor, E_total_supply)
    # utility_j = E_j_supply*(1 - w_j_storage_factor) + R_j_revenue  # utility of selling agent j
    # return utility_j


def allocation_i(E_total_supply,c_i_bidding_price, c_bidding_prices_others):
    E_i_allocation = E_total_supply * (c_i_bidding_price / (c_bidding_prices_others + c_i_bidding_price))
    return E_i_allocation


def calc_gamma():
    return random.uniform(0, 1)
    """This function will ultimately predict storage weight
    This will involve some model predictive AI"""


def buyers_game_optimization(id_buyer, E_i_demand ,supply_on_step, c_macro, bidding_price_i_prev, bidding_prices_all):
    """Level 1 game: distributed optimization"""

    """globally declared variables, do not use somewhere else!!"""
    global E_global_buyers, c_S_global_buyers, c_i_global_buyers, c_l_global_buyers, E_i_demand_global
    E_global_buyers = supply_on_step
    c_S_global_buyers = c_macro[1]
    c_l_global_buyers = bidding_prices_all
    c_i_global_buyers = bidding_price_i_prev
    E_i_demand_global = E_i_demand


    """ Buyers prediction model for defining penalty on gap between demand and allocation,
        depending on future prediction and availability of locally stored energy"""
    alpha = 1 # for now, alpha is just 1

    initial_conditions = [E_global_buyers, E_i_demand_global, c_l_global_buyers, c_i_global_buyers, alpha]

    """ This is a MINIMIZATION of costs"""
    def utility_buyer(x):
        x0 = x[0]       # E_global_buyers
        x1 = x[1]       # E_i_demand_buyers_global
        x2 = x[2]       # c_l_global_buyers
        x3 = x[3]       # c_i_global_buyers               unconstrained
        alpha = x[4]    # prediction weight

        """self designed parametric utility function"""
        """ (E_demand - E_allocation)^n + (E_supply)  + satisfaction of a charged battery!"""
        return alpha * (x1 -  (x0 * (x3/(x2 + x3))))**1.7 + x0 * (x3/(x2 + x3)) * x3

        # """self designed parametric utility function"""
        # # return x4 * x1 - (x0 * (x3 / x2)) * x3 + (x4 - (x0 * (x3 / x2))) * x1
        #
        # """original utility function, minimizes """
        # # return sign*x0*(x3/x2)*x1 - x0*(x3/x2)*x3

    """fix parameters E_global, c_S_global, c_l_global"""
    def constraint_param_x0(x):
        return E_global_buyers - x[0]

    def constraint_param_x1(x):
        return E_i_demand_global - x[1]

    def constraint_param_x2(x):
        return c_l_global_buyers - x[2]

    def constraint_param_x4(x):
        return alpha - x[4]

    """incorporate various constraints"""
    con0 = {'type': 'eq', 'fun': constraint_param_x0}
    con1 = {'type': 'eq', 'fun': constraint_param_x1}
    con2 = {'type': 'eq', 'fun': constraint_param_x2}
    con4 = {'type': 'eq', 'fun': constraint_param_x4}


    cons = [con0, con1, con2, con4]
    bounds_buyer = ( (0, None), (0, None), (0, None), (0, None), (0, None) )

    """optimize using SLSQP(?)"""
    sol_buyer = minimize(utility_buyer, initial_conditions, method='SLSQP', bounds=bounds_buyer, constraints=cons)
    # print("optimization result is a bidding price of %f" % sol.x[3])
    print("buyer %d game results in %s" % (id_buyer, sol_buyer.x[3]))

    """return 4th element of solution vector."""
    return sol_buyer, sol_buyer.x[3]



def sellers_game_noBattery_optimization():
    """ function without w_j is not even an optimization but """
    pass


def calc_wj(E_demand, E_horizon):
    w_j_storage = (E_demand**2/sum(E_horizon**2))
    return w_j_storage

def sellers_game_optimization(id_seller, E_j_seller, R_total, E_total_supply, R_prediction, E_surplus_prediction, w_j_storage_factor):
    """ Anticipation on buyers is plugged in here"""

    global Ej_global_sellers, R_total_global_sellers, E_global_sellers, R_prediction_global, E_prediction_global, wj_global_sellers

    Ej_global_sellers = E_j_seller
    R_total_global_sellers = R_total
    E_global_sellers = E_total_supply
    R_prediction_global = R_prediction
    E_prediction_global = E_surplus_prediction
    wj_global_sellers = w_j_storage_factor

    initial_conditions_seller = [Ej_global_sellers, R_total_global_sellers, E_global_sellers, R_prediction_global, E_prediction_global, wj_global_sellers]

    """ This is a MAXIMIZATION of revenue"""
    def utility_seller(x):

        Ej = x[0]   # Ej_global_sellers
        Rd = x[1]   # R_total_global_sellers
        E  = x[2]   # E_global_sellers
        Rp = x[3]   # R_prediction_global
        Ep = x[4]   # E_prediction_global

        wj = x[5]  # wj_global_sellers       unconstrained


        """New Utility"""
        return Rp * (Ej * (1 - wj)/ (Ep + Ej * (1 - wj))) + Rd * ( Ej * wj / (E + (Ej * wj)))  # E must be made as " everything except j"

        """old utility"""
        # return sign * x0 *(1 - x4) + x2 * x1 * ((x0 * x4) / x3)

    def constraint_param_seller0(x):
        return Ej_global_sellers - x[0]

    def constraint_param_seller1(x):
        return R_total_global_sellers - x[1]

    def constraint_param_seller2(x):
        return E_global_sellers - x[2]

    def constraint_param_seller3(x):
        return R_prediction_global - x[3]

    def constraint_param_seller4(x):
        return E_prediction_global - x[4]

    def constraint_minimum_load():
        """here goes constraints that involve minima/maxima on what-ever the consumer definitely needs"""
        pass

    """incorporate various constraints"""
    con_seller0 = {'type': 'eq', 'fun': constraint_param_seller0}
    con_seller1 = {'type': 'eq', 'fun': constraint_param_seller1}
    con_seller2 = {'type': 'eq', 'fun': constraint_param_seller2}
    con_seller3 = {'type': 'eq', 'fun': constraint_param_seller3}
    con_seller4 = {'type': 'eq', 'fun': constraint_param_seller4}

    cons_seller = [con_seller0, con_seller1, con_seller2, con_seller3, con_seller4]
    bounds_seller = ((0, None), (0, None), (0, None), (0, None), (0, None), (0.001, 0.999))

    sol_seller = minimize(utility_seller, initial_conditions_seller, method='SLSQP', bounds=bounds_seller, constraints=cons_seller)  # bounds=bounds
    print("seller %d game results in %s" % (id_seller, sol_seller.x[5]))

    """return 5th element of solution vector."""
    return sol_seller, sol_seller.x[5]
    pass





"""testing c_i within domain C_i
if c_i_price_vector in possible_c_i:
    print("all fine")
else:
    print("macro-grid is competing")

"""


"""error/test functions"""




