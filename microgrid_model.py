import pandas as pd
import numpy as np
import random

from mesa.time import RandomActivation
from mesa import Agent
from mesa import Model
from function_file import *
import PvModel
import BatteryModel

##############
### SET-UP ###
##############

# generate some noise to generalise the load pattern for multiple houses
noise = np.random.normal(0,1,100)   # 0 is the mean of the normal distribution you are choosing from
                                    # 1 is the standard deviation of the normal distribution
                                    # 100 is the number of elements you get in array noise


c_S = 0                                     # c_S is selling price of the microgrid
c_B = 10                                    # c_B is buying price of the microgrid
C_i = (c_S, c_B)                            # Domain of available prices for bidding player i
possible_c_i = range(C_i[0], C_i[1])        # domain of solution for bidding price c_i


##################
### INPUT DATA ###
##################

# Import household characteristics on generation and battery
fname1 = 'household_char.xlsx'
household_char = pd.read_excel(fname1,
                                   sheet='Sheet1')
household_char_indexed = household_char.set_index("unique_id")        # Sets Dataframe with unique_id as index

# Import load data over time
fname2 = 'load.csv'
load_data = np.genfromtxt(fname2,
                     delimiter =',',
                     usecols = 0,
                     missing_values ='NA',
                     usemask =True)

# Import weather data over time
fname3 = 'weather_2014_nov_9.csv'
data = np.genfromtxt(fname3,
                     delimiter=',',
                     skip_header=1,
                     usecols = np.array([0,10,20]),
                     missing_values='NA',
                     usemask=True)


# Extract weather data
td = data[:, 0]  # time in epoch
td2 = pd.to_datetime(td, unit='s')
td3 = np.array(td2[:-1], dtype=np.datetime64)
wind_speed_data = data[:, 1]  # wind speed
rad_data = data[:, 2]  # irradiance

# Initialize output vectors
simLen = len(load_data)             # length of the data sey
load1 = np.zeros(simLen)            # return an array of zeros of length simLen
pv1_out = np.zeros(simLen)          # return array of same length as simLen init pv1_out
batt1_out = np.zeros(simLen)                                              # init battery1 output
batt1_soc = np.zeros(simLen)                                              # init battery1 soc
net_out = np.zeros(simLen)                                                # init netto power out

# Simulation parameters
dt = 60  # s

# PV model
eta_pv1 = 0.15  # conversion efficiency    eta_pv1 is input argument to pvModel init
S_pv1 = 100  # area in m2
pv1 = PvModel.PvModel(eta_pv1, S_pv1)

# Battery model
Capa_batt1 = 20*1e3  # Wh
maxChargePower_batt1 = -10e3  # W
maxDischargePower_batt1 = 10e3  # W
initSoc_batt1 = 80  # %
batt1 = BatteryModel.BatteryModel(Capa_batt1,
                                  maxChargePower_batt1,
                                  maxDischargePower_batt1,
                                  initSoc_batt1,
                                  dt*1.0/3600)


######################
## CREATE MICROGRID ##
######################


class HouseholdAgent(Agent):

    """All microgrid agents should be generated here; initialisation of prosumer tools """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.id = unique_id
        self.battery_capacity = 0             # random.choice(range(15)) * batterycapacity_total
        self.pv_generation = 0              # random.choice(range(15)) * pvgeneration
        self.consumption = 0              # random.choice(range(15)) * consumption
        self.classification = 'passive'
        self.results = []
        self.supply = 0
        self.demand = 0
        """control variables c_i and w_j"""
        self.bidding_price = 0
        self.w_j_storage_factor = 1

    def step(self, agents, big_data_file_per_step):

        """Agent optimization step, what ever specific agents do on during step"""
        self.consumption = big_data_file_per_step[self.id, 0] # import load file
        self.pv_generation = big_data_file_per_step[self.id, 1] # import generation file
        self.battery_capacity = big_data_file_per_step[self.id, 2]
        pool = define_pool(self.consumption, self.pv_generation)               # First define the pool
        self.classification = pool[0]

        """Define players pool and let agents act accordingly"""
        if self.classification == "buyer":
            """Start buyers game init"""
            # print("<Household %d says: I am a %s" % (self.id, self.classification))
            self.classification = ['buyer']
            self.demand = pool[2]
            self.supply = 0
        if self.classification == "seller":
            """ Sellers game init"""
            # print("<Household %d says: I am a %s" % (self.id, self.classification))
            self.classification = ['seller']
            self.supply = calc_supply(pool[1], self.w_j_storage_factor)  # pool contains classification and supply Ej per agents in this round
            self.demand = 0
        return self.supply, self.classification

    def update_game_variables(self, variable):
        if self.classification == 'buyer':
            self.bidding_price = variable
        if self.classification == 'seller':
            self.w_j_storage_factor = variable

    def __repr__(self):
        return "<ID: %d, BatteryCapacity:%d, PvGeneration:%d, Consumption:%d>" % (self.id, self.battery_capacity, self.pv_generation, self.consumption)


            # def seller_utility(self):
    #     """calculates sellers utility"""
    #     pass
    #
    #
    # def buyer_utility(selfs):
    #     """calculates buyers utility"""
    #     pass



    # def classification(self):
    #     """is this agent a prosumer of a consumer? So leader or follower"""
    #     if self.pv_generation < self.consumption:
    #         self.classification = ["Leader"]
    #     else:
    #         self.classification = ["Follower"]





class MicroGrid(Model):
    """create environment in which agents can operate"""

    def __init__(self, N, big_data_file):

        """Initialization and automatic creation of agents"""
        self.num_households = N
        self.steps = 0
        self.time = 0
        self.big_data_file = big_data_file
        self.agents = []
        self.supply_on_step = 0
        self.bidding_prices_all = 0

        for e in range(self.num_households):    # create a set of N agents with activations schedule and
            agent = HouseholdAgent(e, self)
            self.agents.append(agent)

    def step(self):

        """Environment proceeds a step after all agents took a step."""
        random.shuffle(self.agents)
        sellers_pool = []
        buyers_pool = []
        for agent in self.agents[:]:
            return_agent_step = agent.step(self.agents, self.big_data_file[self.steps])
            self.supply_on_step += return_agent_step[0]                                     # supply_on_step is aggregate total energy on that step available in the microgrid to be either sold or stored
            classification = return_agent_step[1]
            self.bidding_prices_all += agent.bidding_price
            if classification == ['buyer']:
                """Level 1 init game among buyers"""
                agent.update_game_variables(agent.bidding_price)                            # update variable in agent.self
                buyers_pool.append([agent.id, agent.bidding_price])
            elif classification == ['seller']:
                """Level 2 init of game of sellers"""
                agent.update_game_variables(agent.w_j_storage_factor)                       # update variable in agent.self
                sellers_pool.append([agent.id, agent.w_j_storage_factor])

        """Optimization is only possible after all players are devided"""
        for agent in self.agents[:]:
            if classification == ['buyer']:
                print(agent.bidding_price)
                agent.bidding_price = buyers_game_optimization(self.supply_on_step, agent.bidding_price, self.bidding_prices_all, C_i)
            else:
                pass

        for agent in self.agents[:]:
            if classification == ['seller']:
                pass
                agent.w_j_storage_factor = sellers_game_optimization(agent.supply, w_peragent, bidding_price_peragent, c_S, bidding_price_i)
            else:
                pass
        """ Update time """
        self.steps += 1
        self.time += 1
        return self.supply_on_step, buyers_pool, sellers_pool                     # total supply of surplus energy E = E_j*w_j_sharing_factor


