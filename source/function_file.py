import csv
import pdb
import random
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
# from source.plots import *

lambda11 = 2
lambda12 = 1
lambda21 = 2
lambda22 = 2
lambda_set = [lambda11, lambda12, lambda21, lambda22]


""" DATA """
def check_file(filename):
    with open(filename) as csvfile:
        CSVread_check = csv.reader(csvfile, delimiter=';')
        # print(CSVread_check)
        # print(len(list(CSVread_check)))
        return len(list(CSVread_check))


def get_usable():
    """ Determines which files can be used as data-sets to agents"""
    list_usable_folders = np.array([])
    with open('/Users/dirkvandenbiggelaar/Desktop/DATA/list_of_prod_folders.csv') as list_of_prod_folders:
        csv_usable = csv.reader(list_of_prod_folders, delimiter=',')
        for row in csv_usable:
            list_usable_folders = np.append(list_usable_folders, int(row[-1]))
        return list_usable_folders, len(list_usable_folders)



def read_csv_production(filename,duration):
    """ Read in usable data for production of agents"""
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


def read_csv_load(filename,duration):
    """ Read in usable data for load of agents"""
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

    """Interesting definition of surplus; including battery preference"""
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


def calc_supply(surplus_per_agent, w_j_storage_factor):
    supply_per_agent = surplus_per_agent * w_j_storage_factor
    return supply_per_agent


def calc_R_j_revenue(R_total, E_j_supply, w_j_storage_factor, E_total_supply):
    """calculation of revenue for seller j depending on allocation to i"""
    R_j = R_total * (E_j_supply*w_j_storage_factor)/E_total_supply    # payment_i is E_i*ci = amount of demanded energy * bidding price
    return R_j


def get_preferred_soc(soc_preferred, battery_capacity, E_prediction_series, horizon_length):
    """determines the preferred state of charge for given agent/battery """

    future_availability = 0
    for i in range(horizon_length):
        if E_prediction_series[i] < 0.001:
            E_prediction_series[i] = 0.001

        future_availability += E_prediction_series[i]

    """ A.I. future work: horizon_length is exploration reinforcement learning; what is the optimal horizon such that the battery is never fully
        depleted, but is used the most extensive possible for maximising profit. 
        Depleted battery is a punishment / pushing it right to the edge is a reward """

    max_E_predicted = max(E_prediction_series[0:horizon_length])
    average_predicted_E_surplus = future_availability/horizon_length

    """E_prediction_current is now subject to heavy changes, might there be a spike in E_prediction_current, then there is a 
        corresponding spike in soc_preferred, thus maybe make E_prediction_current an average to make it more robust"""
    E_prediction_current = E_prediction_series[0]

    """ Only considering E_predicted and no actual E_demand of the user: only measure abundancy of the resource, not the actual 
    need of the good..."""
    weight_preferred_soc = 0.5**(average_predicted_E_surplus/E_prediction_current)**0.3
    soc_preferred = battery_capacity * weight_preferred_soc

    return soc_preferred

""""""
def get_preferred_soc_new(batt_capacity_agent, soc_preferred, E_prediction):
    """determines the preferred state of charge for given agent/battery"""
    soc_preferred = soc_preferred
    return soc_preferred


def allocation_i(E_total_supply, c_i_bidding_price, c_bidding_prices_others):
    """ calculation of allocation to agent i"""
    if c_i_bidding_price <= 0 or (c_bidding_prices_others + c_i_bidding_price) <= 0:
        E_i_allocation = 0
        return E_i_allocation
    try:
        E_i_allocation = E_total_supply * (c_i_bidding_price / (c_bidding_prices_others + c_i_bidding_price))
    except RuntimeWarning:
        pdb.set_trace()
        E_i_allocation = 0

    return E_i_allocation


def calc_gamma():
    return random.uniform(0, 1)
    """This function will ultimately predict storage weight
    This will involve some model predictive AI"""

def buyers_game_optimization(id_buyer, E_i_demand ,supply_on_step, c_macro, bidding_price_i_prev, bidding_prices_others_opt, E_batt_available, SOC_gap_agent, lambda_set):
    """Level 1 game: distributed optimization"""
    """ https: // stackoverflow.com / questions / 17009774 / quadratic - program - qp - solver - that - only - depends - on - numpy - scipy """
    """globally declared variables, do not use somewhere else!!"""
    def utility_buyer(c_i, E_i_opt, E_j_opt, c_l_opt, lambda11, lambda12):
        """self designed parametric utility function"""
        """ Closing the gap vs costs for closing the gap"""


        # u_i = (abs(E_i_opt - E_j_opt * c_i / (c_l_opt + c_i)))**lambda11 + (c_i * E_j_opt * (c_i/(c_l_opt + c_i)))**lambda12
        # if RuntimeWarning:
        #     return 0
        # return u_i

        try:
            return (abs(E_i_opt - E_j_opt * c_i / (c_l_opt + c_i)))**lambda11 + (c_i * E_j_opt * (c_i/(c_l_opt + c_i)))**lambda12
        except RuntimeWarning:
            return 0




    bounds_buyer = [(0, None)]

    E_i = E_i_demand
    E_j = supply_on_step
    c_l = bidding_prices_others_opt
    soc_av = E_batt_available
    soc_gap = SOC_gap_agent

    init = bidding_price_i_prev

    lambda11 = lambda_set[0]
    lambda12 = lambda_set[1]

    """optimize using SLSQP(?)"""
    sol_buyer = minimize(lambda x : utility_buyer(x, E_i, E_j, c_l, lambda11, lambda12), init, method='SLSQP', bounds=bounds_buyer)

    """return 4th element of solution vector."""
    return sol_buyer, sol_buyer.x[0]

def calc_utility_function_i(E_i_demand, E_total_supply, c_i_bidding_price, c_bidding_prices_others, lambda_set):
    """This function calculates buyers utility"""
    E_i_allocation = allocation_i(E_total_supply, c_i_bidding_price, c_bidding_prices_others)

    lambda11 = lambda_set[0]
    lambda12 = lambda_set[1]
    """ I made demand-gap to be absolute, meaning also higher allocation than demand is a penalty... not more that logical since it
        would be weird to need for 10 and to receive/passively buy 15 """
    utility_i = (abs(E_i_demand - E_i_allocation))**lambda11 + (c_i_bidding_price * E_i_allocation)**lambda12

    demand_gap = abs(E_i_demand - E_i_allocation)
    utility_demand_gap  = (abs(E_i_demand - E_i_allocation))**lambda11
    utility_costs       = (c_i_bidding_price * E_i_allocation)**lambda12

    return utility_i, demand_gap, utility_demand_gap, utility_costs

def sellers_game_optimization(id_seller, E_j_seller, R_direct, E_supply_others, R_prediction, E_supply_prediction, w_j_storage_factor, E_j_prediction_seller, lower_bound_on_w_j, lambda_set):
    """ Anticipation on buyers is plugged in here"""

    """ This is a MAXIMIZATION of revenue"""
    def utility_seller(w, R_p_opt, E_j_opt, E_p_opt, R_d_opt, E_d_opt, E_j_p_opt, lambda21, lambda22):

        """New Utility"""
        u_j =  - ( (R_p_opt * (E_j_p_opt * (1 - w) / (E_p_opt + E_j_p_opt * (1 - w)))) ** lambda21
                       + (R_d_opt * (E_j_opt * w / (E_d_opt + E_j_opt * w))) ** lambda22 )
        return u_j

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
    E_j_p = E_j_prediction_seller

    lambda21 = lambda_set[2]
    lambda22 = lambda_set[3]

    init = w_j_storage_factor

    """  - (R_prediction * (E_j_prediction_seller * (1 - w_j_storage_factor) / (E_supply_prediction + E_j_prediction_seller * (1 - w_j_storage_factor))))**lambda21 
         - (R_direct     * (E_j_seller *                 w_j_storage_factor  / (E_supply_others     + E_j_seller *                 w_j_storage_factor)))**lambda22
    """

    sol_seller = minimize(lambda w : utility_seller(w, R_p, E_j, E_p, R_d, E_d, E_j_p, lambda21, lambda22), init, method='SLSQP', bounds=bounds_seller)

    """return 5th element of solution vector."""
    return sol_seller, sol_seller.x[0], utility_seller(sol_seller.x[0], R_p, E_j, E_p, R_d, E_d, E_j_p, lambda21, lambda22)

def calc_utility_function_j(id_seller, E_j_seller, R_direct, E_supply_others, R_prediction, E_supply_prediction, w_j_storage_factor, E_j_prediction_seller, lambda_set):
    """This function calculates sellers utility"""

    R_p_opt = R_prediction
    E_j_opt = E_j_seller
    w       = w_j_storage_factor
    E_p_opt = E_supply_prediction
    R_d_opt = R_direct
    E_d_opt = E_supply_others
    E_j_p_opt = E_j_prediction_seller
    lambda21 = lambda_set[2]
    lambda22 = lambda_set[3]

    prediction_utility =    - (R_p_opt * (E_j_p_opt * (1 - w) / (E_p_opt + E_j_p_opt * (1 - w)))) ** lambda21
    direct_utility =        - (R_d_opt * (E_j_opt * w / (E_d_opt + E_j_opt * w))) ** lambda22

    utility_j = prediction_utility + direct_utility

    return prediction_utility, direct_utility, utility_j


"""PREDICTION"""
def calc_R_prediction(R_total, big_data_file, horizon, agents, steps):
    """defines alpha weight according to future load - production"""
    gap_per_agent = np.zeros(len(agents))
    gap = np.zeros(horizon)
    E_predicted_per_agent = np.zeros(len(agents))
    E_predicted_total = np.zeros(horizon)

    def weight_filter(horizon, distance):
        """ makes distant predictions weight heavier or lighter"""
        return

    if steps == 220:
        print('220')
        pass

    """calculate gap between load and (local) production"""
    for i in range(horizon):
        # weight_on_value = weight_filter(horizon, distance)
        for agent in agents[:]:
            """ Looks at the demand of each agent in the microgrid """
            gap_per_agent[agent.id] = big_data_file[steps + i][agent.id][0] - big_data_file[steps + i][agent.id][1]
            """ Looks at generation per household in the microgrid """
            E_predicted_per_agent[agent.id] = big_data_file[steps + i][agent.id][1]

        """ absolute value of gap between production and consumption"""
        gap[i] = sum(gap_per_agent)
        if gap[i] < 0:
            gap[i] = 0
        E_predicted_total[i] = sum(E_predicted_per_agent)

    """ alpha looks at the future availability of energy in the system: predicted surplus"""
    try:
        alpha = gap[0]**0.5/(sum(gap**0.5/horizon))
    except RuntimeWarning or sum(gap) == 0 or len(horizon) < 10:
        alpha = 1

    """ beta looks at """
    try:
        beta = E_predicted_total[0]**0.5/(sum(E_predicted_total**0.5)/horizon)
    except RuntimeWarning or sum(E_predicted_total) == 0 or len(horizon) < 10:
        beta = 1

    if math.isnan(beta):
        beta = 1
    if math.isnan(alpha):
        alpha = 1

    """ test which one works better"""
    R_prediction =  (beta * R_total + alpha * R_total)/2

    return R_prediction, alpha, beta

def calc_c_prediction():
    c_prediction = 0.5
    return c_prediction


def calc_R_prediction_masked(R_total, big_data_file, horizon, agents, comm_reach, steps):
    """defines alpha weight according to future load - production"""
    gap_per_agent = np.zeros(len(agents))
    gap = np.zeros(horizon)
    E_predicted_per_agent = np.zeros(len(agents))
    E_predicted_total = np.zeros(horizon)
    E_predicted_per_agent_list_masked = np.zeros(len(comm_reach))

    def weight_filter(horizon, distance):
        """ makes distant predictions weight heavier or lighter"""
        return

    """calculate gap between load and (local) production"""
    for i in range(horizon):
        # weight_on_value = weight_filter(horizon, distance)
        for agent in agents[:]:
            """ Looks at the demand of each agent in the microgrid """
            gap_per_agent[agent.id] = big_data_file[steps + i][agent.id][0] - big_data_file[steps + i][agent.id][1]
            """ Looks at generation per household in the microgrid """
            E_predicted_per_agent[agent.id] = big_data_file[steps + i][agent.id][1]
        for i in range(len(comm_reach)):
            E_predicted_per_agent_list_masked[i] = E_predicted_per_agent[comm_reach[i]]

        """ absolute value of gap between production and consumption"""
        gap[i] = sum(gap_per_agent)
        if gap[i] < 0:
            gap[i] = 0
        E_predicted_total[i] = sum(E_predicted_per_agent_list_masked)

    """ alpha looks at the future availability of energy in the system: predicted surplus"""
    alpha = gap[0]**0.5/((sum(gap)/horizon)**0.5)
    if RuntimeWarning or sum(gap) == 0 or len(horizon) < 10 or gap[0] == 0:
        alpha = 1

    """ beta looks at """
    beta = E_predicted_total[0]**0.5/((sum(E_predicted_total)/horizon)**0.5)
    if RuntimeWarning or sum(E_predicted_total) == 0 or len(horizon) < 10 or E_predicted_total[0] == 0:
        beta = 1

    """ test which one works better"""
    R_prediction =  (beta * R_total + alpha * R_total)/2

    return R_prediction, alpha, beta

def calc_c_prediction():
    c_prediction = 0.5
    return c_prediction


def calc_E_total_prediction(surplus_per_step_prediction, horizon, N, step, prediction_range):
    """ linear weighted prediction for total predicted energy surplus per step"""
    E_total_prediction_step = 0
    for agent in range(N):
        E_total_prediction_step += surplus_per_step_prediction[agent]
    return E_total_prediction_step


def calc_E_surplus_prediction(E_prediction_step, horizon, N, prediction_range, step):
    """ atm this is linear weighted. I would make sense to make closer values more important/heavier weighted,
        then; for faster response it is important to act on shorter-term prediction"""
    E_surplus_prediction = 0
    for steps in range(horizon):
        E_surplus_prediction += E_prediction_step[steps]
    """linear defined E_surplus_prediction_over_horizon"""
    E_surplus_prediction_avg = E_surplus_prediction/horizon
    return E_surplus_prediction_avg


def calc_w_prediction():
    w_prediction = 0.5
    return w_prediction


def calc_wj(E_demand, E_horizon):
    w_j_storage = (E_demand**2/sum(E_horizon**2))
    return w_j_storage
"""DYNAMICS"""

def Peukerts_law():
    """Discharging slower is better/more efficient!"""
    # k = 1.2
    # Capacity_p = (I_discharge**k)*t
    # efficiency_discharge =
    """http://ecee.colorado.edu/~ecen2060/materials/lecture_notes/Battery3.pdf"""
    return


""" Functions specific to Validation algorithm"""

def utility_i_validation(E_opt, c_others_opt, c_i_opt):

    def utility_function_i_validation(c_i, E, c_others):
        return  E * c_i / sum(c_others + c_i) -  (E * c_i / sum(c_others + c_i)) * c_i

    bounds_buyer = 0,1
    init = c_i_opt
    sol_i_validation= minimize(lambda c : utility_function_i_validation(c_i_opt, E_opt, c_others_opt), init, method='SLSQP', bounds=bounds_buyer)  # bounds=bounds, constraints=cons_seller

    return sol_i_validation



def utility_j_validation(E_j_opt, gamma_j_opt, R_j_opt, w_j_val):

    def utility_function_j_validation(w_j, E_j, gamma_j, R, E):

        return np.log(1 + E_j * (1 - w_j)) + gamma_j * R * E_j * w_j / E

    bounds_seller = 0,1
    init = w_j_val
    sol_j_validation= minimize(lambda w_val : utility_function_j_validation(w_val, E_j_opt, gamma_j_opt, R_j_opt), init, method='SLSQP', bounds=bounds_seller)  # bounds=bounds, constraints=cons_seller

    return sol_j_validation, sol_j_validation.x[4]


def isNaN(num):
    return num != num





