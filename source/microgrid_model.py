""" 2 paradigm concept where sellers determine w_j according their own prediction models and always supply full demand:= no game during times of plenty"""

from source.BatteryModel import *
from source.PvModel import *
import pandas as pd
from source.function_file import *
from source.initialization import *
from mesa import Agent, Model


import ctypes
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
duration = 200  # 1440                # Duration of the sim (one full day or 1440 time_steps of 1 minute) !!10 if test!!
N = 3                           # N agents only
step_list = np.zeros([duration])

c_S = 10                                              # c_S is selling price of the microgrid
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
        self.w_j_storage_factor = 0
        self.E_j_supply = 0
        self.c_i_bidding_price = 0
        self.stored = 0
        self.bidding_prices_others = 0
        self.E_others_supply = 0
        """prediction"""
        self.horizon_agent = 2                                          # should not be bigger than total durations duhh
        self.E_surplus_prediction = np.zeros(self.horizon_agent)
        self.w_j_prediction = 0.5
        """Battery related"""
        self.battery_capacity_n = 10000
        self.soc_preferred = 0
        self.soc_actual = 5
        self.soc_gap = 0
        self.battery_influx = 0
        self.batt_available = 0

        """results"""
        self.results = []
        self.tolerance_seller = 1
        self.tolerance_buyer = 1
        self.tol_buyer = []


    def step(self, big_data_file_per_step, big_data_file, steps):
        """Agent optimization step, what ever specific agents do on during step"""
        """real time data"""
        self.consumption = big_data_file_per_step[self.id, 0]           # import load file
        self.pv_generation = big_data_file_per_step[self.id, 1]         # import generation file
        self.battery_capacity = big_data_file_per_step[self.id, 2]      # maximum amount
        self.available_storage = self.battery_capacity + self.stored_energy - self.used_energy
        """define buyers/sellers classification"""
        [classification, surplus_agent, demand_agent] = define_pool(self.consumption, self.pv_generation)
        self.classification = classification

        """ agents personal prediction"""
        for steps in range(self.horizon_agent):
            self.E_surplus_prediction = big_data_file[steps][self.id][0] - big_data_file[steps][self.id][1]  # load - production

        """Determine state of charge of agent's battery"""
        self.soc_preferred = get_preferred_soc(self.battery_capacity_n)
        self.soc_gap = self.soc_preferred - self.soc_actual
        if self.soc_gap < 0:
            self.soc_gap = 0
        self.soc_actual += self.battery_influx

        charge_rate = 100

        """Define players pool and let agents act accordingly"""
        if self.classification == 'buyer':
            """buyers game init"""
            self.E_i_demand = demand_agent + (self.soc_preferred - self.soc_actual)/charge_rate
            self.batt_available = self.battery_capacity_n - self.soc_actual
            if self.E_i_demand < 0:
                """ if buyer has battery charge left (after preferred_soc), it can either 
                store this energy and do nothing, waiting until this surplus has gone,"""
                self.classification = 'passive'
                self.soc_actual -= self.demand_agent
                """ or start to play the selling game with it"""
                # self.classification = 'seller'
                # self.E_j_surplus = abs(self.E_i_demand)
            else:
                self.c_i_bidding_price = random.uniform(min(c_macro), max(c_macro))
                print("Household %d says: I am a %s" % (self.id, self.classification))
                """values for sellers are set to zero"""
                self.E_j_surplus = 0
                self.w_j_storage_factor = 0
                self.gamma = 0
        elif self.classification == 'seller':
            """sellers game init"""
            self.E_j_surplus = surplus_agent
            self.w_j_storage_factor = random.uniform(0.2, 0.8)
            self.E_j_supply = calc_supply(surplus_agent, self.w_j_storage_factor)  # pool contains classification and E_j_surplus Ej per agents in this round
            print("Household %d says: I am a %s" % (self.id, self.classification))
            """values for seller are set to zero"""
            self.c_i_bidding_price = 0
            self.E_i_demand = 0
            self.E_i_allocation = 0
            self.payment_to_seller = 0
        else:
            print("Household %d says: I am a %s agent" % (self.id, self.classification))

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
        self.E_total_surplus = 0
        self.c_bidding_prices = np.zeros(N)
        self.c_nominal = 0
        self.w_storage_factors = np.zeros(N)
        self.R_total = 0
        self.E_supply_per_agent = np.zeros(N)
        """Battery"""
        self.E_total_stored = 0
        self.E_total_surplus = 0
        self.battery_capacity_total = 0
        """Unrelated"""
        self.sellers_pool = []
        self.buyers_pool = []
        self.passive_pool = []
        """Two pool prediction defining future load and supply"""
        self.R_prediction = []
        self.E_prediction = []
        self.E_supply_prediction = 0
        self.E_surplus_prediction = 0
        self.w_prediction = 0.5
        self.c_nominal_prediction = 0
        self.E_demand_prediction = 0
        self.E_surplus_prediction_per_agent = 0
        """create a set of N agents with activations schedule and e = unique id"""
        for e in range(self.num_households):
            agent = HouseholdAgent(e, self)
            self.agents.append(agent)

    def step(self):

        """Environment proceeds a step after all agents took a step"""
        # random.shuffle(self.agents)
        self.sellers_pool = []
        self.buyers_pool = []
        self.passive_pool = []
        self.E_total_supply = 0

        """ Measure """

        """Take initial """
        for agent in self.agents[:]:
            agent.step(self.big_data_file[self.steps], self.big_data_file, self.steps)
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
                self.E_total_surplus += agent.E_j_surplus                               # does not change by optimization
                self.E_total_supply += agent.E_j_surplus * agent.w_j_storage_factor     # does change by optimization
            else:
                self.passive_pool.append(agent.id)

        for agent in self.agents[:]:
            agent.E_others_supply = self.E_total_supply - ( agent.E_j_surplus * agent.w_j_storage_factor )


        """Optimization after division of players into pools"""
        tolerance_global = np.zeros(N)
        tolerance_sellers = np.zeros(N)
        tolerance_buyers = np.zeros(N)

        iteration_global = 0
        iteration_buyers = 0
        iteration_sellers = 0
        self.c_nominal = 0
        self.R_total = 0  # resets for every step

        """Global optimization"""
        initial_values = self.c_bidding_prices
        payment_to_seller = np.zeros(N)
        """global level"""
        while True: # global
            iteration_global += 1
            self.c_nominal = sum(self.c_bidding_prices, 0) / len(self.buyers_pool) # not good
            prev_c_nominal = self.c_nominal
            print("total energy available to buyers is ", self.E_total_supply)
            """Buyers level optimization"""
            while True: # buyers
                iteration_buyers += 1
                """agent.c_i_bidding_price should incorporate its E_i_demand. If allocation is lower than the actual E_i_demand, payment to macro-grid 
                (which will be more expensive than buying from Alice) needs to be minimised. 
                Utility of buyer should be: (total energy demand - the part allocated from alice(c_i) ) * c_macro
                then allocation will be increased by offering more money"""

                for agent in self.agents[:]:
                    if agent.classification == 'buyer':
                        """preparation for update"""
                        prev_bid = agent.c_i_bidding_price
                        agent.bidding_prices_others = bidding_prices_others(self.c_bidding_prices, agent.c_i_bidding_price)
                        agent.E_i_allocation = allocation_i(self.E_total_supply, agent.c_i_bidding_price, agent.bidding_prices_others)
                        utility_i, demand_gap = calc_utility_function_i(agent.E_i_demand, self.E_total_supply, agent.c_i_bidding_price, agent.bidding_prices_others)
                        # print("OLD: buyer", agent.id, "demand from macro-grid =", agent.E_i_demand - agent.E_i_allocation)
                        # print("OLD: utility buyer", agent.id, "=", utility_i)
                        # agent.E_i_demand =
                        """update c_i"""
                        sol_buyer, sol_buyer.x[3] = buyers_game_optimization(agent.id, agent.E_i_demand, self.E_total_supply, c_macro, agent.c_i_bidding_price, agent.bidding_prices_others, agent.batt_available, agent.soc_gap)
                        agent.c_i_bidding_price =  sol_buyer.x[3]
                        self.c_bidding_prices[agent.id] = agent.c_i_bidding_price
                        agent.bidding_prices_others = bidding_prices_others(self.c_bidding_prices, agent.c_i_bidding_price)
                        agent.E_i_allocation = allocation_i(self.E_total_supply, agent.c_i_bidding_price, agent.bidding_prices_others)
                        # print("new allocation to buyer %d = %f" % (agent.id, agent.E_i_allocation))
                        payment_to_seller[agent.id] = agent.c_i_bidding_price * agent.E_i_allocation
                        # print("NEW: buyer", agent.id, "demand from macro-grid =", agent.E_i_demand - agent.E_i_allocation)
                        # print("NEW: utility buyer", agent.id, "=", utility_i)
                        new_bid = agent.c_i_bidding_price
                        agent.tol_buyer.append(new_bid - prev_bid)
                        # print("tol_buyer ",agent.id,"= ",agent.tol_buyer)
                        tolerance_buyers[agent.id] = abs(new_bid - prev_bid)
                    else:
                        self.c_bidding_prices[agent.id] = 0  # current total payment offered


                epsilon_buyers_game = max(abs(tolerance_buyers))
                if epsilon_buyers_game < 0.01:
                    """ Values  to be plugged into sellers game"""
                    self.c_nominal = sum(self.c_bidding_prices, 0) / len(self.buyers_pool)
                    self.R_total = sum(payment_to_seller)
                    print("optimization of buyers-level game %d has been completed with e = %f and c_nominal = %f!" % (self.steps, epsilon_buyers_game, self.c_nominal))
                    tolerance_buyers[:] = 0
                    for agent in self.agents[:]:
                        agent.tol_buyer = []
                    break
                else:
                    pass

            """Determine global tolerances"""
            new_c_nominal = self.c_nominal
            tolerance_c_nominal = abs(new_c_nominal - prev_c_nominal)

            """create horizon prediction model"""
            surplus_per_step_prediction = np.zeros((duration, len(self.agents)))
            demand_per_step_prediction = np.zeros((duration, len(self.agents)))

            """Index prediction"""
            for i in range(duration):
                for agent in self.agents[:]:
                    """ now using actual data (big_data_file) but should be substituted with prediction data"""
                    energy_per_step_per_agent_prediction = self.big_data_file[i][agent.id][0] - self.big_data_file[i][agent.id][1]         # [0] = load, corresponds with demand  - [1] = production

                    if energy_per_step_per_agent_prediction <= 0:
                        demand_per_step_prediction[i][agent.id] = abs(energy_per_step_per_agent_prediction)

                    if energy_per_step_per_agent_prediction < 0:
                        surplus_per_step_prediction[i][agent.id] = abs(energy_per_step_per_agent_prediction)

            # print(demand_per_step_prediction)  # elements in list represent total demand in the system for 1 step
            # print(surplus_per_step_prediction) #

            """ for now this is the prediction, horizon is now 1 step ahead"""
            self.E_demand_prediction = sum(demand_per_step_prediction[self.steps:(self.steps + 2)][i],i)    # quadratic

            horizon = 2 # including current step
            self.R_prediction, alpha = calc_R_prediction(self.R_total, self.big_data_file, horizon, self.agents)
            self.E_surplus_prediction_per_agent = surplus_per_step_prediction[self.steps + 1]
            self.E_surplus_prediction = sum(self.E_surplus_prediction_per_agent)
            self.E_supply_prediction = self.E_surplus_prediction * self.w_prediction
            # i is range (out of duration), j is agents
            # big_data_file[1][j][0] = test_load_file_agents[j][i]  # *(random.uniform(0.9, 1.2))
            # big_data_file[1][j][1] = test_production_file_agents[j][i]  # *(random.uniform(0.9, 1.2))

            supply_old = self.E_total_supply

            """sellers-level game optimization"""
            while True: # sellers
                iteration_sellers += 1

                for agent in self.agents[:]:
                    agent.E_others_supply = self.E_total_supply - (agent.E_j_surplus * agent.w_j_storage_factor)
                    agent.bidding_prices_others = sum(self.c_bidding_prices) - agent.c_i_bidding_price
                    agent.E_supply_others_prediction = self.E_supply_prediction - agent.E_surplus_prediction * agent.w_j_prediction #+ self.battery_capacity_total - agent.battery_capacity_n
                for agent in self.agents[:]:
                    if agent.classification == 'seller':
                        """Sellers optimization game, plugging in bidding price to decide on sharing factor.
                        Is bidding price lower than that the smart-meter expects to sell on a later time period?
                        smart-meter needs a prediction on the coming day. Either use the load data or make a predicted model on 
                        all aggregated load data"""
                        prev_wj = agent.w_j_storage_factor
                        agent.bidding_prices_others = sum(self.c_bidding_prices)
                        old_prediction_utility, old_direct_utility, old_utility_j = calc_utility_function_j(agent.id, agent.E_j_surplus, self.R_total, agent.E_others_supply, self.R_prediction, agent.E_supply_others_prediction, agent.w_j_storage_factor)
                        # print("old utility of seller", agent.id, "=", old_utility_j)
                        # print("alpha =", alpha)

                        """update"""
                        sol_seller, sol_seller.x[5] = sellers_game_optimization(agent.id, agent.E_j_surplus, self.R_total, agent.E_others_supply, self.R_prediction, agent.E_supply_others_prediction, agent.w_j_storage_factor)
                        agent.w_j_storage_factor = sol_seller.x[5]


                        self.E_supply_per_agent[agent.id] = agent.E_j_surplus * agent.w_j_storage_factor
                        self.E_total_supply = sum(self.E_supply_per_agent)
                        agent.E_others_supply = self.E_total_supply - (agent.E_j_surplus * agent.w_j_storage_factor)

                        difference_rev = self.R_total - self.R_prediction
                        difference_load = self.E_total_supply - self.E_surplus_prediction
                        # print("diff in revenue =", difference_rev, "diff in load =", difference_load)

                        prediction_utility = self.R_prediction * (agent.E_j_surplus * (1 - agent.w_j_storage_factor) / (self.E_surplus_prediction + agent.E_j_surplus * (1 - agent.w_j_storage_factor)))
                        direct_utility = self.R_total * (agent.E_j_surplus * agent.w_j_storage_factor / (self.E_total_supply + (agent.E_j_surplus * agent.w_j_storage_factor)))
                        # print("prediction_utility =", prediction_utility, "direct_utility =", direct_utility)
                        self.w_storage_factors[agent.id] = agent.w_j_storage_factor

                        prediction_utility, direct_utility, utility_j = calc_utility_function_j(agent.id, agent.E_j_surplus, self.R_total, self.E_total_supply, self.R_prediction, self.E_surplus_prediction, agent.w_j_storage_factor)
                        # print("utility of seller", agent.id, "=", utility_j)

                        new_wj = agent.w_j_storage_factor
                        tolerance_sellers[agent.id] = abs(new_wj - prev_wj)
                        # print("intermittent supply to buyers is =", self.E_total_supply)
                    else:
                        agent.w_j_storage_factor = 0


                self.E_total_supply = 0
                for agent in self.agents[:]:
                    self.E_total_supply += calc_supply(agent.pv_generation, agent.w_j_storage_factor)
                for agent in self.agents[:]:
                    agent.E_others_supply = self.E_total_supply - (agent.E_j_surplus * agent.w_j_storage_factor)

                actual_batteries = np.zeros(N)
                epsilon_sellers_game = max(abs(tolerance_sellers))
                if epsilon_sellers_game < 0.01:
                    print("optimization of sellers-level game %d has been completed with e = %f! in %d rounds" % (self.steps, epsilon_sellers_game, iteration_sellers))
                    print("total surplus = %f, supply to buyers = %f" % (self.E_total_surplus, self.E_total_supply))
                    tolerance_sellers[:] = 0
                    supply_new = self.E_total_supply
                    for agent in self.agents[:]:
                        actual_batteries[agent.id] = agent.soc_actual
                    break
                else:
                    pass

            tolerance_supply = abs(supply_new - supply_old)
            epsilon_c_nominal = abs(tolerance_c_nominal)
            if (epsilon_c_nominal < 0.1 and tolerance_supply < 0.1) or iteration_global > 10 or self.steps > 2:
                print("optimization of round %d has been completed in %d iterations with c_n epsilon = %f!!" % (self.steps, iteration_global, tolerance_supply))
                """settle all deals"""
                for agent in self.agents[:]:
                    if agent.classification == 'buyer':
                        agent.E_i_allocation = allocation_i(self.E_total_supply, agent.c_i_bidding_price, agent.bidding_prices_others)
                        agent.soc_influx = agent.E_i_allocation - agent.E_i_demand
                        agent.soc_actual += agent.soc_influx
                    elif agent.classification == 'seller':
                        agent.soc_influx = agent.E_j_surplus  * agent.w_j_storage_factor
                        agent.soc_actual += agent.soc_influx
                    actual_batteries[agent.id] = agent.soc_actual
                    print("actual_batteries =", actual_batteries)

                break
            else:
                pass


        """ This is still vague """
        for agent in self.agents[:]:
            """Battery capacity after step, coded regardless of classification"""
            load_covering = 0  # how much of stored energy is needed to cover total load, difficult!
            agent.stored += agent.E_j_surplus * (1 - agent.w_j_storage_factor) + agent.E_i_allocation - (agent.E_j_surplus*agent.w_j_storage_factor + load_covering)
            self.E_total_stored += agent.stored

        """ Update time """
        self.steps += 1
        self.time += 1
        return self.E_total_surplus, self.E_total_supply, self.buyers_pool, self.sellers_pool
