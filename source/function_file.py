import csv
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
# from source.plots import *

"""GLOABAL optimization variable"""




global lambda1, lambda2, lambda11
lambda11 = 0.5
lambda12 = 3
lambda2 = 2

""" DATA """

def check_file(filename):
    with open(filename) as csvfile:
        CSVread_check = csv.reader(csvfile, delimiter=';')
        # print(CSVread_check)
        # print(len(list(CSVread_check)))
        return len(list(CSVread_check))


def get_usable():
    list_usable_folders = np.array([])
    with open('/Users/dirkvandenbiggelaar/Desktop/DATA/list_of_prod_folders.csv') as list_of_prod_folders:
        csv_usable = csv.reader(list_of_prod_folders, delimiter=',')
        for row in csv_usable:
            list_usable_folders = np.append(list_usable_folders, int(row[-1]))
        return list_usable_folders, len(list_usable_folders)


def read_csv_production(filename,duration):                                    # still using a single load pattern for all houses
    """Reads in load and generation data from data set"""
    with open(filename) as csvfile:
        CSVread = csv.reader(csvfile, delimiter=',')
        data_return = np.zeros(300)
        for row in CSVread:
            data_value = float(row[-1])
            if data_value < 0.0:
                data_value = abs(data_value)
            data_return = np.append(data_return, data_value)
        zero = 0.0
        while len(data_return) < duration:
            data_return = np.append(data_return, zero)
        return data_return


def read_csv_load(filename,duration):                                    # still using a single load pattern for all houses
    """Reads in load and generation data from data set"""
    with open(filename) as csvfile:
        CSVread = csv.reader(csvfile, delimiter=',')
        data_return = np.array([])
        for row in CSVread:
            data_value = float(row[-1])
            if data_value < 0.0:
                data_value = 0.0
            data_return = np.append(data_return, data_value)
        while len(data_return) < duration:
            data_return = np.append(data_return, 0)
        return data_return



"""READ BACK IN BIG_DATA_FILE"""
def read_csv_big_data(filename, duration):
    with open(filename) as csvfile:
        CSVread = csv.reader(csvfile, delimiter=',')
        data_return = np.array([])
        for row in CSVread:
            data_value = float(row[-1])
        data_return = np.append(data_return, data_value)
        return data_return

""" ALGORITHM """

def define_pool(consumption_at_round, production_at_round, soc_gap, soc_surplus, charge_rate, discharge_rate):
    """this function has to decide whether agent is a buyer or a seller"""
    surplus_per_agent = (production_at_round + soc_surplus/discharge_rate) - (consumption_at_round + soc_gap/charge_rate)
    if surplus_per_agent > 0:
        classification = 'seller'
        demand_agent = 0
    elif surplus_per_agent < 0:
        classification = 'buyer'
        demand_agent = abs(surplus_per_agent)
        surplus_per_agent = 0
    else:
        classification = 'passive'
        demand_agent = 0
        surplus_per_agent = 0
    return [classification, surplus_per_agent, demand_agent]


def bidding_prices_others(c_bidding_prices, c_i_bidding_price):
    bidding_prices_others_def = sum(c_bidding_prices) - c_i_bidding_price
    return bidding_prices_others_def


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
    utility_i = (E_i_demand - E_i_allocation)**lambda11 + (c_i_bidding_price * E_i_allocation)**lambda12
    demand_gap = E_i_demand - E_i_allocation
    return utility_i, demand_gap


def calc_utility_function_j(id, E_j_surplus, R_direct, E_supply_others, R_prediction, E_prediction_others, w_j_storage_factor):
    """This function calculates sellers utility"""
    direct_utility = - (R_direct * (E_j_surplus * w_j_storage_factor / (E_supply_others + (E_j_surplus * w_j_storage_factor))))**lambda2
    prediction_utility = - (R_prediction * (E_j_surplus * (1 - w_j_storage_factor) / (E_prediction_others + E_j_surplus * (1 - w_j_storage_factor))))**lambda2
    utility_j = prediction_utility + direct_utility
    return prediction_utility, direct_utility, utility_j
    # return  - Rd * (Ej * wj / (E_others + (Ej * wj)))**0.6 - Rp * (Ej * (1 - wj) / (Ep + Ej * (1 - wj))) ** 0.6


def get_preferred_soc(batt_capacity_agent, soc_preferred):
    """determines the preferred state of charge for given agent/battery"""
    soc_preferred = soc_preferred
    return soc_preferred


def allocation_i(E_total_supply,c_i_bidding_price, c_bidding_prices_others):
    E_i_allocation = E_total_supply * (c_i_bidding_price / (c_bidding_prices_others + c_i_bidding_price))
    return E_i_allocation


def calc_gamma():
    return random.uniform(0, 1)
    """This function will ultimately predict storage weight
    This will involve some model predictive AI"""


def buyers_game_optimization(id_buyer, E_i_demand ,supply_on_step, c_macro, bidding_price_i_prev, bidding_prices_others_opt, E_batt_available, SOC_gap_agent):
    """Level 1 game: distributed optimization"""

    """globally declared variables, do not use somewhere else!!"""
    # global E_global_buyers, c_S_global_buyers, c_i_global_buyers, c_l_global_buyers, E_i_demand_global
    #
    # E_global_buyers = supply_on_step
    # c_S_global_buyers = c_macro[1]
    #
    # c_l_global_buyers = bidding_prices_others_opt
    # c_i_global_buyers = bidding_price_i_prev
    #
    # E_i_demand_global = E_i_demand
    #
    # """purely for constraints on max demand"""
    # E_batt_available_global = E_batt_available
    # SOC_gap_global = SOC_gap_agent


    """ Buyers prediction model for defining penalty on gap between demand and allocation,
        depending on future prediction and availability of locally stored energy"""
    # beta = 1 # for now, alpha is just 1

    # initial_conditions = [E_global_buyers, E_i_demand_global, c_l_global_buyers, c_i_global_buyers, beta, SOC_gap_global]
    # constants = [supply_on_step, E_i_demand, bidding_prices_others_opt, SOC_gap_agent]
    """ This is a MINIMIZATION of costs"""
    def utility_buyer(c_i, E_i_opt, E_j_opt, c_l_opt):
        # x0 = x[0]       # E_global_buyers
        # x1 = x[1]       # E_i_demand_buyers_global
        # x2 = x[2]       # c_l_global_buyers
        # x3 = x[3]       # c_i_global_buyers               unconstrained
        # beta = x[4]     # prediction weight
        # soc_gap = x[5]  # SOC_gap_agent

        """self designed parametric utility function"""
        """ Closing the gap vs costs for closing the gap"""
        return (E_i_opt - E_j_opt * c_i / (c_l_opt + c_i)) + (c_i * E_j_opt * (c_i/(c_l_opt + c_i)))**lambda12

        # return (x1 - (x0 * (x3/(x2 + x3))))**lambda11 + (x3 * x0 * (x3/(x2 + x3)))**lambda12

        # """self designed parametric utility function"""
        # # return x4 * x1 - (x0 * (x3 / x2)) * x3 + (x4 - (x0 * (x3 / x2))) * x1
        #
        # """original utility function, minimizes """
        # # return sign*x0*(x3/x2)*x1 - x0*(x3/x2)*x3

    """fix parameters E_global, c_S_global, c_l_global"""
    # def constraint_param_x0(x):
    #     return E_global_buyers - x[0]
    #
    # def constraint_param_x1(x):
    #     return E_i_demand_global - x[1]
    #
    # def constraint_param_x2(x):
    #     return c_l_global_buyers - x[2]
    #
    # def constraint_param_x4(x):
    #     return beta - x[4]
    #
    # def constraint_possible_storage(x):
    #     return (x[0] * (x[3]/(c_l_global_buyers + x[3]))) - E_batt_available_global

    """incorporate various constraints"""
    # con0 = {'type': 'eq', 'fun': constraint_param_x0}
    # con1 = {'type': 'eq', 'fun': constraint_param_x1}
    # con2 = {'type': 'eq', 'fun': constraint_param_x2}
    # con4 = {'type': 'eq', 'fun': constraint_param_x4}
    # con_max_batt = {'type': 'ineq', 'fun': constraint_possible_storage}
    #
    # x0 = x[0]  # E_global_buyers
    # x1 = x[1]  # E_i_demand_buyers_global
    # x2 = x[2]  # c_l_global_buyers
    # x3 = x[3]  # c_i_global_buyers               unconstrained
    # beta = x[4]  # prediction weight
    # soc_gap = x[5]  # SOC_gap_agent

    # cons = [con0, con1, con2, con4, con_max_batt]
    bounds_buyer = [(0, None)]

    E_i = E_i_demand
    E_j = supply_on_step
    c_l = bidding_prices_others_opt
    soc_av = E_batt_available
    soc_gap = SOC_gap_agent

    init = bidding_price_i_prev

    """optimize using SLSQP(?)"""
    sol_buyer = minimize(lambda x : utility_buyer(x, E_i, E_j, c_l), init, method='SLSQP', bounds=bounds_buyer)

    # sol_buyer = minimize(utility_buyer, initial_conditions, method='SLSQP', bounds=bounds_buyer, constraints=cons)
    # print("optimization result is a bidding price of %f" % sol.x[3])
    # print("buyer %d game results in %s" % (id_buyer, sol_buyer.x[3]))

    """return 4th element of solution vector."""
    return sol_buyer, sol_buyer.x[0]


def sellers_game_noBattery_optimization():
    """ function without w_j is not even an optimization but """
    pass


def calc_wj(E_demand, E_horizon):
    w_j_storage = (E_demand**2/sum(E_horizon**2))
    return w_j_storage


def calc_R_prediction(R_total, big_data_file, horizon, agents, steps):
    """defines alpha weight according to future load - production"""
    gap_per_agent = np.zeros(len(agents))
    gap = np.zeros(horizon)
    E_predicted_per_agent = np.zeros(len(agents))
    E_predicted_on_step = np.zeros(horizon)

    """calculate gap between load and (local) production"""
    for i in range(horizon):
        for agent in agents[:]:
            gap_per_agent[agent.id] = big_data_file[steps + i][agent.id][1] - big_data_file[steps + i][agent.id][0]

            E_predicted_per_agent[agent.id] = big_data_file[i][agent.id][0]

        gap[i] = abs(sum(gap_per_agent))
        E_predicted_on_step[i] = sum(E_predicted_per_agent)

    alpha = gap[0]**0.5/(sum(gap**0.5/horizon))
    beta = E_predicted_on_step[0]**0.5/(sum(E_predicted_on_step**0.5/horizon))
    R_prediction = alpha * beta * R_total
    # print("[alpha, beta] = ", [alpha, beta])

    return R_prediction, alpha


def sellers_game_optimization(id_seller, E_j_seller, R_direct, E_supply_others, R_prediction, E_supply_prediction, w_j_storage_factor, lower_bound_on_w_j):
    """ Anticipation on buyers is plugged in here"""

    # global Ej_global_sellers, R_total_global_sellers, E_global_sellers, R_prediction_global, E_prediction_global, wj_global_sellers
    #
    # Ej_global_sellers = E_j_seller
    # R_total_global_sellers = R_total
    # E_global_sellers = E_supply_others
    # R_prediction_global = R_prediction
    # E_prediction_global = E_supply_prediction
    # wj_global_sellers = w_j_storage_factor

    # initial_conditions_seller = [Ej_global_sellers, R_total_global_sellers, E_global_sellers, R_prediction_global, E_prediction_global, wj_global_sellers]

    """ This is a MAXIMIZATION of revenue"""
    def utility_seller(w, R_p_opt, E_j_opt, E_p_opt, R_d_opt, E_d_opt):

        # Ej = x[0]   # Ej_global_sellers
        # Rd = x[1]   # R_total_global_sellers
        # E  = x[2]   # E_global_sellers
        # Rp = x[3]   # R_prediction_global
        # Ep = x[4]   # E_prediction_global
        #
        # wj = x[5]  # wj_global_sellers       unconstrained


        """New Utility"""
        # return Rp  * (1-wj)**2 + Rd * wj**2  # E must be made as " everything except j"


        return - (R_p_opt * (E_j_opt * (1 - w) / (E_p_opt + E_j_opt * (1 - w))))**lambda2- (R_d_opt * (E_j_opt * w/(E_d_opt + E_j_opt*w)))**lambda2

        """old utility"""
        # return sign * x0 *(1 - x4) + x2 * x1 * ((x0 * x4) / x3)

    # def constraint_param_seller0(x):
    #     return Ej_global_sellers - x[0]
    #
    # def constraint_param_seller1(x):
    #     return R_total_global_sellers - x[1]
    #
    # def constraint_param_seller2(x):
    #     return E_global_sellers - x[2]
    #
    # def constraint_param_seller3(x):
    #     return R_prediction_global - x[3]
    #
    # def constraint_param_seller4(x):
    #     return E_prediction_global - x[4]
    #
    # def constraint_minimum_load():
    #     """here goes constraints that involve minima/maxima on what-ever the consumer definitely needs"""
    #     pass

    # """incorporate various constraints"""
    # con_seller0 = {'type': 'eq', 'fun': constraint_param_seller0}
    # con_seller1 = {'type': 'eq', 'fun': constraint_param_seller1}
    # con_seller2 = {'type': 'eq', 'fun': constraint_param_seller2}
    # con_seller3 = {'type': 'eq', 'fun': constraint_param_seller3}
    # con_seller4 = {'type': 'eq', 'fun': constraint_param_seller4}
    #
    # cons_seller = [con_seller0, con_seller1, con_seller2, con_seller3, con_seller4]

    bounds_seller = [(lower_bound_on_w_j, 1.0)]


    R_p = R_prediction
    E_j = E_j_seller
    E_p = E_supply_prediction
    R_d = R_direct
    E_d = E_supply_others


    init = w_j_storage_factor

    sol_seller = minimize(lambda w : utility_seller(w, R_p, E_j, E_p, R_d, E_d), init, method='SLSQP', bounds=bounds_seller)  # bounds=bounds, constraints=cons_seller
    # print("seller %d game results in %s" % (id_seller, sol_seller.x[5]))

    """return 5th element of solution vector."""
    return sol_seller, sol_seller.x[0]


"""PREDICTION"""

def calc_E_prediction(surplus_per_step_prediction, prediction_horizon, N, step, prediction_range):
    E_predicted_per_step = 0
    for agent_id in range(N):
        E_predicted_per_step += surplus_per_step_prediction[agent_id]
    return E_predicted_per_step


"""DYNAMICS"""

def Peukerts_law():
    """Discharging slower is better/more efficient!"""
    # k = 1.2
    # Capacity_p = (I_discharge**k)*t
    # efficiency_discharge =
    """http://ecee.colorado.edu/~ecen2060/materials/lecture_notes/Battery3.pdf"""
    return



"""testing c_i within domain C_i
if c_i_price_vector in possible_c_i:
    print("all fine")
else:
    print("macro-grid is competing")

"""


"""error/test functions"""




