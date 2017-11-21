from source.BatteryModel import *
from source.PvModel import *
import pandas as pd
from source.function_file import *
from source.initialization import *
from mesa import Agent, Model
import math
import scipy.optimize

##############
### SET-UP ###
##############

""" noise: 0 is the mean of the normal distribution you are choosing from
           1 is the standard deviation of the normal distribution
           100 is the number of elements you get in array noise    """
noise = np.random.normal(0, 1, 100)


"""All starting parameters are initialised"""
duration = 4  # 1440                # Duration of the sim (one full day or 1440 time_steps of 1 minute) !!10 if test!!
N = 5                               # N agents only
step_list = np.zeros([duration])

c_S = 4                                              # c_S is selling price of the microgrid
c_B = 1                                              # c_B is buying price of the microgrid
c_macro = (c_B, c_S)                                 # Domain of available prices for bidding player i
possible_c_i = range(c_macro[0], c_macro[1])         # domain of solution for bidding price c_i

"""Battery and PV panel dynamics are hidden here"""
# ##################
# ### INPUT DATA ###
# ##################
#
# # # Import household characteristics on generation and battery
# # fname1 = 'household_char.xlsx'
# # household_char = pd.read_excel(fname1,
# #                                    sheet='Sheet1')
# # household_char_indexed = household_char.set_index("unique_id")        # Sets Dataframe with unique_id as index
#
# # Import load data over time
# fname2 = 'data/load.csv'
# load_data = np.genfromtxt(fname2,
#                      delimiter =',',
#                      usecols = 0,
#                      missing_values ='NA',
#                      usemask =True)
#
# # Import weather data over time
# fname3 = 'data/weather_2014_nov_9.csv'
# data = np.genfromtxt(fname3,
#                      delimiter=',',
#                      skip_header=1,
#                      usecols = np.array([0,10,20]),
#                      missing_values='NA',
#                      usemask=True)
#
#
# # Extract weather data
# td = data[:, 0]  # time in epoch
# td2 = pd.to_datetime(td, unit='s')
# td3 = np.array(td2[:-1], dtype=np.datetime64)
# wind_speed_data = data[:, 1]  # wind speed
# rad_data = data[:, 2]  # irradiance
#
# # Initialize output vectors
# simLen = len(load_data)             # length of the data sey
# load1 = np.zeros(simLen)            # return an array of zeros of length simLen
# pv1_out = np.zeros(simLen)          # return array of same length as simLen init pv1_out
# batt1_out = np.zeros(simLen)                                              # init battery1 output
# batt1_soc = np.zeros(simLen)                                              # init battery1 soc
# net_out = np.zeros(simLen)                                                # init netto power out
#
# # Simulation parameters
# dt = 60  # s
#
# # PV model
# eta_pv1 = 0.15  # conversion efficiency    eta_pv1 is input argument to pvModel init
# S_pv1 = 100  # area in m2
# pv1 = PvModel(eta_pv1, S_pv1)
#
# # Battery model
# Capa_batt1 = 20*1e3  # Wh
# maxChargePower_batt1 = -10e3  # W
# maxDischargePower_batt1 = 10e3  # W
# initSoc_batt1 = 80  # %
# batt1 = BatteryModel(Capa_batt1,
#                                   maxChargePower_batt1,
#                                   maxDischargePower_batt1,
#                                   initSoc_batt1,
#                                   dt * 1.0 / 3600)
#

######################
## CREATE MICROGRID ##
######################


class HouseholdAgent(Agent):

    """All microgrid household(agents) should be generated here; initialisation of prosumer tools """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        """agent characteristics"""
        self.id = unique_id
        self.battery_capacity = 10                              # everyone has 10 kwh battery
        self.pv_generation = random.uniform(0, 1)               # random.choice(range(15)) * pvgeneration
        self.consumption = random.uniform(0, 1)                 # random.choice(range(15)) * consumption
        self.classification = []

        """control variables"""
        self.gamma = initialize("gamma")                                 # sellers
        self.E_j_surplus = initialize("E_j_surplus")                    # sellers
        self.E_i_demand = initialize("E_i_demand")                      # buyers
        self.used_energy = initialize("used_energy")                        # both
        self.E_i_allocation = initialize("E_i_allocation")                  # buyers
        self.stored_energy = initialize("stored_energy")                # both
        self.available_storage = initialize("available_storage")        # both
        self.payment_to_seller = initialize("payment_to_seller")        # buyer
        self.w_j_storage_factor = initialize("w_j_storage_factor")
        self.E_j_supply = 0
        self.c_i_bidding_price = 0
        self.stored = 0

        """results"""
        self.results = []

    def step(self, big_data_file_per_step):
        """Agent optimization step, what ever specific agents do on during step"""
        """real time data"""
        self.consumption = big_data_file_per_step[self.id, 0]           # import load file
        self.pv_generation = big_data_file_per_step[self.id, 1]         # import generation file
        self.battery_capacity = big_data_file_per_step[self.id, 2]      # maximum amount
        self.available_storage = self.battery_capacity + self.stored_energy - self.used_energy
        """define buyers/sellers classification"""
        [classification, surplus_agent, demand_agent] = define_pool(self.consumption, self.pv_generation)
        self.classification = classification

        """Define players pool and let agents act accordingly"""
        if self.classification == 'buyer':
            """buyers game init"""
            self.E_i_demand = demand_agent
            self.c_i_bidding_price = random.uniform(min(c_macro), max(c_macro))
            print("Household %d says: I am a %s" % (self.id, self.classification))
            """values for sellers are set to zero"""
            self.E_j_surplus = 0
            self.w_j_storage_factor = 0
            self.gamma = 0
        elif self.classification == 'seller':
            """sellers game init"""
            self.E_j_surplus = surplus_agent
            self.E_j_supply = calc_supply(surplus_agent, self.w_j_storage_factor)  # pool contains classification and E_j_surplus Ej per agents in this round
            print("Household %d says: I am a %s" % (self.id, self.classification))
            """values for seller are set to zero"""
            self.c_i_bidding_price = 0
            self.E_i_demand = 0
            self.E_i_allocation = 0
            self.payment_to_seller = 0
        return


    def __repr__(self):
        return "ID: %d, batterycapacity:%d, pvgeneration:%d, consumption:%d" % (self.id, self.battery_capacity, self.pv_generation, self.consumption)


"""Microgrid model environment"""


class MicroGrid(Model):
    """create environment in which agents can operate"""

    def __init__(self, N, big_data_file):

        """Initialization and automatic creation of agents"""
        self.num_households = N
        self.steps = 0
        self.time = 0
        self.big_data_file = big_data_file
        self.agents = []
        self.E_total_supply = 0
        self.c_bidding_prices = np.zeros(N)
        self.c_nominal = 0
        self.w_storage_factors = np.zeros(N)
        self.R_total = 0

        """Battery"""
        self.E_total_stored = 0

        """Unrelated crap"""
        self.sellers_pool = []
        self.buyers_pool = []




        """create a set of N agents with activations schedule and e = unique id"""
        for e in range(self.num_households):
            agent = HouseholdAgent(e, self)
            self.agents.append(agent)

    def step(self):

        """Environment proceeds a step after all agents took a step"""
        # random.shuffle(self.agents)
        self.sellers_pool = []
        self.buyers_pool = []
        self.E_total_supply = 0

        """Take initial """
        for agent in self.agents[:]:
            agent.step(self.big_data_file[self.steps])
            classification = agent.classification
            if classification == 'buyer':
                """Level 1 init game among buyers"""
                self.buyers_pool.append(agent.id)
                self.c_bidding_prices[agent.id] = agent.c_i_bidding_price
                agent.w_j_storage_factor = 0
            elif classification == 'seller':
                """Level 2 init of game of sellers"""
                self.sellers_pool.append(agent.id)
                self.w_storage_factors[agent.id] = agent.w_j_storage_factor
                self.E_total_supply += agent.E_j_surplus * agent.w_j_storage_factor

        # print(self.buyers_pool, self.sellers_pool)

        """Optimization after division of players into pools"""
        tolerance_global = 1
        tolerance_buyers = 1
        iteration_global = 0
        iteration_buyers = 0
        self.c_nominal = 0
        self.R_total = 0  # resets for every step

        """Global optimization"""
        while abs(tolerance_global) > 0.001:
            iteration_global += 1
            """Buyers level optimization"""
            while abs(tolerance_buyers) > 0.001:
                iteration_buyers += 1
                prev_nominal_price = self.c_nominal
                """agent.c_i_bidding_price should incorporate its E_i_demand. If allocation is lower than the actual E_i_demand, payment to macro-grid 
                (which will be more expensive than buying from Alice) needs to be minimised. 
                Utility of buyer should be: (total energy demand - the part allocated from alice(c_i) ) * c_macro
                then allocation will be increased by offering more money"""

                for agent in self.agents[:]:
                    if agent.classification == 'buyer':
                        prev_bid = agent.c_i_bidding_price
                        bidding_prices_summed = sum(self.c_bidding_prices)
                        agent.E_i_allocation = allocation_i(self.E_total_supply, agent.c_i_bidding_price, bidding_prices_summed)
                        utility_i = calc_utility_function_i(self.E_total_supply, c_macro, agent.c_i_bidding_price, bidding_prices_summed)
                        print("buyer", agent.id, "demand from macro-grid =", agent.E_i_demand - agent.E_i_allocation)
                        print("utility buyer", agent.id, "=", utility_i)
                        """update"""
                        sol_buyer, sol_buyer.x[3] = buyers_game_optimization(agent.id, agent.E_i_demand, self.E_total_supply, c_macro, agent.c_i_bidding_price, bidding_prices_summed)
                        agent.c_i_bidding_price =  sol_buyer.x[3]
                        self.c_bidding_prices[agent.id] = agent.c_i_bidding_price
                        bidding_prices_summed = sum(self.c_bidding_prices)
                        agent.E_i_allocation = allocation_to_i_func(self.E_total_supply, agent.c_i_bidding_price, bidding_prices_summed)
                        agent.payment_to_seller = agent.c_i_bidding_price * agent.E_i_allocation
                        self.R_total += agent.payment_to_seller
                        print("buyer", agent.id, "demand from macro-grid =", agent.E_i_demand - agent.E_i_allocation)
                        print("utility buyer", agent.id, "=", utility_i)

                    else:
                        self.c_bidding_prices[agent.id] = 0  # current total payment offered

                self.c_nominal = sum(self.c_bidding_prices, 0) / len(self.buyers_pool)
                tolerance_buyers = prev_nominal_price - self.c_nominal

            """sellers-level game optimization"""
            """Sellers optimization game, plugging in bidding price to decide on sharing factor.
                Is bidding price lower than that the smart-meter expects to sell on a later time period?
                smart-meter needs a prediction on the coming day. Either use the load data or make a predicted model on 
                all aggregated load data"""

            new_supply_agents = 0  # resets for every step
            old_wj_vector =  self.w_storage_factors# self.w_storage_factors

            """ A diminising return function is needed. Lets make it quadratic thenn"""
            for agent in self.agents[:]:
                if agent.classification == 'seller':
                    agent.gamma = calc_gamma()

                    sol_seller, sol_seller.x[4] = sellers_game_optimization(agent.id, self.R_total, agent.E_j_supply, self.E_total_supply, agent.gamma, agent.w_j_storage_factor)

                    # guessvalue = 0.5
                    # fsolve_sellers = scipy.optimize.fsolve(lambda x: (1 + agent.E_j_supply * (1 - x) + self.R_total * (agent.E_j_supply * x / self.E_total_supply)), guessvalue)[0]  # 5 is guess value
                    # print('fsolve_sellers', fsolve_sellers)

                    agent.w_j_storage_factor = sol_seller.x[4]
                    self.w_storage_factors[agent.id] = agent.w_j_storage_factor
                    new_supply_agents += calc_supply(agent.pv_generation, agent.w_j_storage_factor)
                    utility_j = calc_utility_function_j(agent.id, self.R_total, agent.E_j_supply, agent.w_j_storage_factor, self.E_total_supply)
                    print("utility of seller", agent.id, "=", utility_j)

            new_wj_vector = self.w_storage_factors
            tolerance_global = abs(max(np.subtract(new_wj_vector, old_wj_vector)))
            self.E_total_supply = new_supply_agents

        for agent in self.agents[:]:
            """Battery capacity after step, coded regardless of classification"""
            load_covering = 0  # how much of stored energy is needed to cover total load, difficult!
            agent.stored += agent.E_j_surplus * (1 - agent.w_j_storage_factor) + agent.E_i_allocation - (agent.E_j_surplus*agent.w_j_storage_factor + load_covering)
            self.E_total_stored += agent.stored

        """ Update time """
        self.steps += 1
        self.time += 1
        return self.E_total_supply, self.buyers_pool, self.sellers_pool
