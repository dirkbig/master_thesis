import csv
import pdb
import random
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


# """"""""""""""""""""""""""
# """ PREDICTION ON/OFF  """
""""""""""""""""""""""""""
#prediction = 'off'
prediction = 'on'


""""""""""""""""""""""""""""""
""" SAT CONSTRAINT ON/OFF  """
""""""""""""""""""""""""""""""
constraints = 'off'
# constraints = 'on'


factor_revenue = 1

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


def get_preferred_soc(soc_preferred, battery_capacity, E_prediction_series, soc_actual, horizon_length, predicted_E_consumption_list):
    """determines the preferred state of charge for given agent/battery """

    agent = 'prosumer'
    future_load = 0
    future_surplus = 0
    for i in range(horizon_length):
        if E_prediction_series[i] < 0.0001:
            E_prediction_series[i] = 0.0001
        future_load += predicted_E_consumption_list[i]
        future_surplus += E_prediction_series[i]

    if np.all(E_prediction_series <= 0.01):
        """ this agent can be considered a consumer and not a producer at this stage"""
        agent = 'consumer'


    if agent == 'prosumer':
        """ A.I. future work: horizon_length is exploration reinforcement learning; what is the optimal horizon such that the battery is never fully
            depleted, but is used the most extensive possible for maximising profit. 
            Depleted battery is a punishment / pushing it right to the edge is a reward """
        # max_E_predicted = max(E_prediction_series[0:horizon_length])
        average_predicted_E_surplus = future_surplus/horizon_length

        """E_prediction_current is now subject to heavy changes, might there be a spike in E_prediction_current, then there is a 
            corresponding spike in soc_preferred, thus maybe make E_prediction_current an average to make it more robust"""
        E_prediction_current = E_prediction_series[0]

        """ Only considering E_predicted and no actual E_demand of the user: only measure abundancy of the resource, not the actual 
        need of the good..."""
        weight_preferred_soc = 0.5**(average_predicted_E_surplus/E_prediction_current)**0.3
        soc_preferred = battery_capacity * weight_preferred_soc

    elif agent == 'consumer':
        """ strategy for (temporary) consumers """
        E_consumption_current = predicted_E_consumption_list[0]
        average_consumption_over_horizon = future_load/horizon_length
        weight_preferred_soc_consumer = 0.5**(average_consumption_over_horizon/E_consumption_current)**0.3

        soc_preferred = battery_capacity * weight_preferred_soc_consumer


    """ load shedding """
    if soc_actual > 0.8 * battery_capacity:
        soc_preferred = 0.2 * battery_capacity

    if soc_actual < 0.2 * battery_capacity:
        soc_preferred = 0.8 * battery_capacity

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
    """This function will ultimately predict storage weight
    This will involve some model predictive AI"""
    return random.uniform(0, 1)


def buyers_game_optimization(id_buyer, E_i_demand ,supply_on_step, c_macro, bidding_price_i_prev, bidding_prices_others_opt, E_batt_available, SOC_gap_agent, lambda_set, actuator_saturation_ESS_charge):
    """Level 1 game: distributed optimization"""

    def utility_buyer(c_i, E_i_opt, E_j_opt, c_l_opt, lambda11, lambda12):
        """self designed parametric utility function, Closing the gap vs costs for closing the gap"""
        try:
            return (abs(E_i_opt - E_j_opt * c_i / (c_l_opt + c_i)))**lambda11 + (c_i * E_j_opt * (c_i/(c_l_opt + c_i)))**lambda12
        except RuntimeWarning:
            return 0

    E_i = E_i_demand
    E_j = supply_on_step
    c_l = bidding_prices_others_opt
    bounds_buyer = [(0, None)]
    init = bidding_price_i_prev

    lambda11 = lambda_set[0]
    lambda12 = lambda_set[1]

    """ Actuator saturation on charging """
    if constraints == 'on':
        cons_sat = {'type': 'ineq',
                    'fun': lambda c_i: np.array(actuator_saturation_ESS_charge -  E_j * c_i / (c_l + c_i)),
                    'jac': lambda c_i: np.array((E_i * c_l) / ((c_l + c_i)**2))}
        """optimize using SLSQP(?)"""
        sol_buyer = minimize(lambda x: utility_buyer(x, E_i, E_j, c_l, lambda11, lambda12), init, method='SLSQP',
                             constraints=cons_sat, bounds=bounds_buyer)

    else:
        """optimize using SLSQP(?)"""
        sol_buyer = minimize(lambda x: utility_buyer(x, E_i, E_j, c_l, lambda11, lambda12), init, method='SLSQP',
                                                bounds=bounds_buyer)



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



def sellers_game_optimization(id_seller, E_j_seller, R_direct, E_supply_others, R_prediction, E_supply_prediction, w_j_storage_factor, E_j_prediction_seller, lower_bound_on_w_j, lambda_set, actuator_saturation_ESS_discharge):
    """ Anticipation on buyers is plugged in here"""

    """ This is a MAXIMIZATION of revenue"""
    def utility_seller(w, R_p_opt, E_j_opt, E_p_opt, R_d_opt, E_d_opt, E_j_p_opt, lambda21, lambda22):

        """ Utility function seller with prediction decision """
        return - ( (R_p_opt * (E_j_p_opt * (1 - w) / (E_p_opt + E_j_p_opt * (1 - w)))) ** lambda21
                       + (R_d_opt * (E_j_opt * w / (E_d_opt + E_j_opt * w))) ** lambda22 )


    R_p = R_prediction
    E_j = E_j_seller
    E_p = E_supply_prediction
    R_d = R_direct
    E_d = E_supply_others
    E_j_p = E_j_prediction_seller

    bounds_seller = [(min(lower_bound_on_w_j, 0.99), 1.0)]
    init = w_j_storage_factor
    lambda21 = lambda_set[2]
    lambda22 = lambda_set[3]

    """ Actuator saturation on discharging """
    if constraints == 'on':
        cons_sat = {'type': 'ineq',
                    'fun': lambda w: np.array(actuator_saturation_ESS_discharge - E_j * (1-w)),
                    'jac': lambda w: E_j}
        sol_seller = minimize(lambda w: utility_seller(w, R_p, E_j, E_p, R_d, E_d, E_j_p, lambda21, lambda22), init,
                              method='SLSQP', constraints=cons_sat, bounds=bounds_seller)
    else:
        sol_seller = minimize(lambda w: utility_seller(w, R_p, E_j, E_p, R_d, E_d, E_j_p, lambda21, lambda22), init,
                              method='SLSQP',  bounds=bounds_seller)

    return sol_seller, sol_seller.x[0], utility_seller(sol_seller.x[0], R_p, E_j, E_p, R_d, E_d, E_j_p, lambda21, lambda22)


def sellers_game_optimization_no_prediction(id_seller, E_j_seller, R_direct, E_supply_others, R_prediction, E_supply_prediction, w_j_storage_factor, E_j_prediction_seller, lower_bound_on_w_j, lambda_set, actuator_saturation_ESS_discharge):
    """ Anticipation on buyers is plugged in here"""
    factor_revenue = 0.4

    def utility_seller_no_prediction(w, R_p_opt, E_j_opt, E_p_opt, R_d_opt, E_d_opt, E_j_p_opt, lambda21, lambda22):
        """ Utility function seller without prediction decision"""
        return np.log(1 + E_j_opt * (1 - w)) + factor_revenue * (R_d_opt * (E_j_opt * w / (E_d_opt + E_j_opt * w)))

    R_p = R_prediction
    E_j = E_j_seller
    E_p = E_supply_prediction
    R_d = R_direct
    E_d = E_supply_others
    E_j_p = E_j_prediction_seller

    bounds_seller = [(min(lower_bound_on_w_j, 0.99), 1.0)]
    init = w_j_storage_factor
    lambda21 = lambda_set[2]
    lambda22 = lambda_set[3]

    """ Actuator saturation on charging """
    if constraints == 'on':
        cons_sat = {'type': 'ineq',
                    'fun': lambda w: np.array(actuator_saturation_ESS_discharge - E_j * (1-w)),
                    'jac': lambda w: E_j}
        sol_seller = minimize(
            lambda w: utility_seller_no_prediction(w, R_p, E_j, E_p, R_d, E_d, E_j_p, lambda21, lambda22), init,
            method='SLSQP', constraints=cons_sat, bounds=bounds_seller)

    else:
        sol_seller = minimize(lambda w: utility_seller_no_prediction(w, R_p, E_j, E_p, R_d, E_d, E_j_p, lambda21, lambda22), init,
            method='SLSQP', bounds=bounds_seller)


    return sol_seller, sol_seller.x[0], utility_seller_no_prediction(sol_seller.x[0], R_p, E_j, E_p, R_d, E_d, E_j_p, lambda21, lambda22)


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

    if prediction == 'on':
        prediction_utility =    - (R_p_opt * (E_j_p_opt * (1 - w) / (E_p_opt + E_j_p_opt * (1 - w)))) ** lambda21
        direct_utility =        - (R_d_opt * (E_j_opt * w / (E_d_opt + E_j_opt * w))) ** lambda22
        utility_j = prediction_utility + direct_utility

    if prediction == 'off':
        utility_j =   np.log(1 + E_j_opt * (1 - w)) + factor_revenue * (R_d_opt * (E_j_opt * w / (E_d_opt + E_j_opt * w)))
        prediction_utility = None
        direct_utility = None

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
    horizon_factor_up = np.arange(0.5, 1.5, prediction_range)
    horizon_factor_down = np.arange(1.5, 0.5, prediction_range)

    factor = min(horizon_factor_up, horizon_factor_down)

    for steps in range(horizon):
        E_surplus_prediction += E_prediction_step[steps]
    """linear defined E_surplus_prediction_over_horizon"""
    E_surplus_prediction_avg = E_surplus_prediction/horizon
    return E_surplus_prediction_avg


def calc_w_prediction():
    w_prediction = 0.8 #sine_constant + 0.7 * np.sin(np.pi * f * i / Fs + 0.25 * np.pi)
    return w_prediction


def calc_wj(E_demand, E_horizon):
    w_j_storage = (E_demand**2/sum(E_horizon**2))
    return w_j_storage


"""GRID DYNAMICS"""
def Peukerts_law():
    """Discharging slower is better/more efficient!"""
    # k = 1.2
    # Capacity_p = (I_discharge**k)*t
    # efficiency_discharge =
    """http://ecee.colorado.edu/~ecen2060/materials/lecture_notes/Battery3.pdf"""
    return

def get_PV_satuation(step_time):

    return

def get_ESS_satuation(step_time):
    """ Actuator saturation for supplying or recieving energy in kWh"""
    P_max_charge    = (0.2 * 15) / ((60 / step_time))
    P_max_discharge = (0.2 * 15) / ((60 / step_time))

    return P_max_discharge, P_max_charge



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



def prediction_base_function(R_total, big_data_file, horizon, prediction_range, agents, N, steps):
    """ Prediction base function"""

    """ surplus_prediction/demand_per_step_prediction gives predicted surplus/demand; per step, per agent"""
    surplus_prediction = np.zeros((prediction_range, N))
    demand_per_step_prediction = np.zeros((prediction_range, N))
    E_surplus_prediction_list = np.zeros(N)
    E_supply_prediction_list = np.zeros(N)
    """ index prediction data, now using the current data set... AI should be plugged in here eventually"""
    for i in range(prediction_range):
        for agent in agents[:]:
            """ now using actual data (big_data_file) but should be substituted with prediction data"""
            energy_per_step_per_agent_prediction = big_data_file[steps + i][agent.id][0] - \
                                                   big_data_file[steps + i][agent.id][
                                                       1]  # [0] = load, corresponds with demand  - [1] = production
            """ results in either demand_per_step_prediction or surplus_prediction for each agent"""
            if energy_per_step_per_agent_prediction >= 0:
                """ series of coming demands"""
                demand_per_step_prediction[i][agent.id] = abs(energy_per_step_per_agent_prediction)
            if energy_per_step_per_agent_prediction < 0:
                """ series of coming surpluses"""
                surplus_prediction[i][agent.id] = abs(energy_per_step_per_agent_prediction)

    """ for now this is the prediction """
    E_total_surplus_prediction_per_step = np.zeros(
        prediction_range)  # must adapt in size for every step (shrinking as time progresses))
    E_total_demand_prediction = np.zeros(prediction_range)

    for i in range(prediction_range):
        E_total_surplus_prediction_per_step[i] = calc_E_total_prediction(surplus_prediction[i][:], horizon, N,
                                                                              steps,
                                                                              prediction_range)
    """ A prediction"""
    R_prediction, alpha, beta = calc_R_prediction(R_total, big_data_file, horizon, agents,
                                                       steps)

    """ E_surplus prediction over horizon in total = a sum of agents predictions"""
    E_surplus_prediction_over_horizon = 0
    for agent in agents[:]:
        E_surplus_prediction_over_horizon += agent.E_prediction_agent
        E_surplus_prediction_list[agent.id] = agent.E_prediction_agent
    """ conversion to usable E_supply_prediction (using w_prediction_avg does nothing yet)
        Make use of agents knowledge that is shared among each others: 
        total predicted energy = sum(individual predictions)"""
    w_prediction_avg = 0
    for agent in agents[:]:
        w_prediction_avg += agent.w_prediction / N
        E_supply_prediction_list[agent.id] = E_surplus_prediction_list[agent.id] * agent.w_prediction

    E_supply_prediction = E_surplus_prediction_over_horizon * w_prediction_avg
    E_supply_prediction_list = E_surplus_prediction_list * w_prediction_avg
    """ analysis of prediction data """
    means_surplus = []
    means_load = []

    for i in range(prediction_range):
        means_surplus.append(np.mean(surplus_prediction[i][:]))
        means_load.append(np.mean(demand_per_step_prediction[i][:]))

    return R_prediction, E_supply_prediction, E_supply_prediction_list, w_prediction_avg

def isNaN(num):
    return num != num










""" Costfunction of generation; PSO general"""
def costfunction(x, *args):
    """ Costfunction """
    """ agents specific weights"""
    alpha_vector_pso = args[0]
    beta_vector_pso = args[1]
    gamma_vector_pso = args[2]
    cost = sum(alpha_vector_pso) + np.dot(beta_vector_pso, x) + np.dot(gamma_vector_pso, x ** 2)
    return cost

""" PSO validation HIERARCHICAL"""
def buyer_objective_function_PSO(c, args_buyer):
    """ Costfunction Buyer PSO"""
    cost_buyers = 0

    """ agents specific weights"""
    E_i = args_buyer[0]
    E_j_opt = args_buyer[1]
    lambda11_arg= args_buyer[2]
    lambda12_arg = args_buyer[3]
    N = args_buyer[4]
    c_l_opt = sum(c)

    for i in range(N):
        E_i_opt = E_i[i] # demand per agent (buying)
        _lambda11 = lambda11_arg[i]
        _lambda12 = lambda12_arg[i]


        cost_buyers += abs(E_i_opt - E_j_opt * c[i] / (c_l_opt + c[i])) ** _lambda11 + (c[i] * E_j_opt * (c[i] / (c_l_opt + c[i]))) ** _lambda12

    return cost_buyers

def seller_objective_function_PSO(w, args_seller):
    """ Costfunction Sellers PSO"""
    cost_sellers = 0

    """ agents specific weights"""
    R_direct = args_seller[0]           # predicted future profit
    E_surplus_list = args_seller[1]     # agents surplus energy
    R_future_list = args_seller[2]      # deterministic direct profit
    E_predicted_list = args_seller[3]   # predicted future surplus energy

    lambda21 = args_seller[4]
    lambda22 = args_seller[5]
    N_seller = args_seller[6]

    for j in range(N_seller):
        E_surplus_j = E_surplus_list[j]
        R_future_j = R_future_list[j]
        E_predicted_j = E_predicted_list[j]

        _lambda21 = lambda21[j]
        _lambda22 = lambda22[j]

        cost_sellers += - ( (R_future_j * (E_surplus_j * (1 - w[j]) / (E_predicted_j + E_surplus_j * (1 - w[j])))) ** _lambda21 + \
                            (R_direct * (E_surplus_j * w[j] / (np.dot(w, E_surplus_list)))) ** _lambda22 )

    return cost_sellers

def combined_objective_function_PSO(x, *args):
    N_mixed, args_buyer, args_seller = args
    N = N_mixed[0]
    N_buyers = N_mixed[1]
    N_sellers = N_mixed[2]
    c = x[0:N_buyers]
    w = x[-N_sellers:N]

    cost_buyers = buyer_objective_function_PSO(c, args_buyer)
    cost_sellers = seller_objective_function_PSO(w, args_seller)
    cost_sellers = abs(cost_sellers)

    cost_total = cost_buyers + cost_sellers

    return cost_total
