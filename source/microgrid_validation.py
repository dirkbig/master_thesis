""" Validation model: algorithm of Distributed Energy Trading in Microgrids: A Game-Theoretic Model and
    Its Equilibrium Analysis

    NO prediction or batteries """

from source.function_file import *
from source.plots import *
import sys

import pandas as pd
import numpy as np
from mesa import Agent, Model

"""All starting parameters are initialised, creating a same structure """
starting_point = 0
stopping_point = 7000 - starting_point - 2000
step_day = 1440
timestep = 5
days = 5


step_time = 10
total_steps = step_day*days
sim_steps = int(total_steps/step_time)

N = 7                            # N agents only
step_list = np.zeros([sim_steps])

c_S = 10                                             # c_S is selling price of the microgrid
c_B = 1                                              # c_B is buying price of the microgrid
c_macro = (c_B, c_S)                                 # Domain of available prices for bidding player i
possible_c_i = range(c_macro[0], c_macro[1])         # domain of solution for bidding price c_i


class HouseholdAgent_val(Agent):

    """All microgrid household(agents) should be generated here; initialisation of prosumer tools """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        """agent characteristics"""
        self.id = unique_id
        self.battery_capacity_n = 3000                         # every household has an identical battery, for now
        self.pv_generation = random.uniform(0, 1)               # random.choice(range(15)) * pvgeneration
        self.consumption = random.uniform(0, 1)                 # random.choice(range(15)) * consumption
        self.classification = []

        """control variables"""
        self.E_j_surplus = 0                                            # sellers
        self.E_j_supply = 0
        self.E_i_demand = 0                                             # buyers
        self.E_i_allocation = 0                                         # buyers
        self.payment_to_seller = 0                                      # buyer
        self.w_j_storage_factor = 0
        self.E_j_supply = 0
        self.c_i_bidding_price = 0
        self.stored = 0
        self.bidding_prices_others = 0
        self.E_supply_others = 0
        self.w_nominal = 0

        """ utilities of agents"""
        self.utility_i = 0
        self.utility_j = 0

        """ merely a summary """
        self.utilities_seller = [0, 0, 0]
        self.utilities_buyer = [0, 0, 0, 0]

        """prediction"""
        self.current_step = 0
        self.max_horizon = 50
        self.horizon_agent = min(self.max_horizon, sim_steps - self.current_step)  # including current step
        self.predicted_E_surplus_list = np.zeros(self.horizon_agent)
        self.w_j_prediction = 0.5
        """Battery related"""
        self.soc_preferred =   500
        self.soc_actual = 500
        self.soc_gap = 0
        self.soc_influx = 0
        self.batt_available = 0
        self.soc_surplus = 0
        self.charge_rate = 0
        self.discharge_rate = 0
        self.lower_bound_on_w_j = 0
        self.deficit = 0
        """results"""
        self.results = []
        self.tolerance_seller = 1
        self.tolerance_buyer = 1
        self.tol_buyer = []
        self.action = []

        self.E_prediction_agent = 0
        self.w_prediction = 0


    def step(self, big_data_file_per_step, big_data_file, E_total_surplus_prediction_per_step, horizon, prediction_range, steps):           # big_data_file = np.zeros((N, step_time, 3))
        """Agent optimization step, what ever specific agents do on during step"""








class MicroGrid_val(Model):
    """create environment in which agents can operate"""

    def __init__(self, N, big_data_file, starting_point):

        """Initialization and automatic creation of agents"""
        self.num_households = N
        self.steps = 0
        self.time = 0
        self.big_data_file = big_data_file
        self.agents = []
        self.E_total_supply = 0
        self.E_total_surplus = 0
        self.c_bidding_prices = np.zeros(N)
        self.c_nominal = 0
        self.w_storage_factors = np.zeros(N)
        self.R_total = 0
        self.E_supply_per_agent = np.zeros(N)
        self.E_demand = 0
        self.E_allocation_per_agent = np.zeros(N)
        self.E_total_supply_list = np.zeros(N)
        """Battery"""
        self.E_total_stored = 0
        self.E_total_surplus = 0
        self.battery_soc_total = 0
        self.actual_batteries = np.zeros(N)
        """Unrelated"""
        self.sellers_pool = []
        self.buyers_pool = []
        self.passive_pool = []
        """Two pool prediction defining future load and supply"""
        self.R_prediction = 0
        self.E_prediction = 0
        self.E_supply_prediction = 0
        self.E_surplus_prediction_over_horizon = 0
        self.w_prediction_avg = 0.5
        self.c_nominal_prediction = 0
        self.E_total_demand_prediction = 0
        self.E_surplus_prediction_per_agent = 0

        """ prediction range is total amount of steps left """
        self.prediction_range = sim_steps - self.steps  # data ends at end of day
        if self.prediction_range <= 0:
            self.prediction_range = 1

        self.E_total_surplus_prediction_per_step = np.zeros(self.prediction_range)

        self.w_nominal = 0
        self.utilities_sellers = np.zeros((N, 3))
        self.utilities_buyers = np.zeros((N, 4))

        self.soc_preferred = np.zeros(N)

        """create a set of N agents with activations schedule and e = unique id"""
        for e in range(self.num_households):
            agent = HouseholdAgent_val(e, self)
            self.agents.append(agent)


    def step(self):

        return