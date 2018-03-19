from blockchain.smartcontract import *
from functions.function_file import *
import sys

import numpy as np
from mesa import Agent, Model



batt_low = 5
batt_high = 15
range_parameter_sweep = 10
interval = (batt_high - batt_low) / range_parameter_sweep

battery_range = np.arange(batt_low, batt_high, interval)



print(battery_range)




# latency_list = np.array([1,0,0])
#
#
# id_non_lagging_agents = np.flatnonzero(latency_list == 0)
#
#
# print(id_non_lagging_agents)
#











#
# ##########################
# ### ASYNCHRONOUS MODEL ###
# ##########################
#
#
# """"""""""""""""""""""""""
# """ BLOCKCHAIN ON/OFF  """
# """"""""""""""""""""""""""
# blockchain = 'off'
#
# blockchain = 'off'
#
# """"""""""""""""""""""""""""""""""""""""""""
# """ TRADING ON/NO-PREDICTION/SUPPLY-ALL  """
# """"""""""""""""""""""""""""""""""""""""""""
#
# prediction = 'on'
# # prediction = 'off'
#
#
#
#
#
#
#
#
# """ noise: 0 is the mean of the normal distribution you are choosing from
#            1 is the standard deviation of the normal distribution
#            100 is the number of elements you get in array noise    """
# noise = np.random.normal(0, 1, 100)
#
# """All starting parameters are initialised"""
# starting_point = 0
# stopping_point = 7200 - starting_point - 2000
# step_day = 1440
# days = 5
# blockchain = 'off'
#
# step_time = 100
# total_steps = step_day*days
# sim_steps = int(total_steps/step_time)
#
# step_list = np.zeros([sim_steps])
#
# c_S = 10                                             # c_S is selling price of the microgrid
# c_B = 1                                              # c_B is buying price of the microgrid
# c_macro = (c_B, c_S)                                 # Domain of available prices for bidding player i
# possible_c_i = range(c_macro[0], c_macro[1])         # domain of solution for bidding price c_i
#
# e_buyers = 0.001
# e_sellers = 0.001
# e_global = 0.01
# e_cn = 0.01
# e_supply = 0.1
#
# class HouseholdAgent(Agent):
#
#     """All microgrid household(agents) should be generated here; initialisation of prosumer tools """
#     def __init__(self, unique_id, model, w3, addr, agents_set, N, comm_radius):
#         super().__init__(unique_id, model)
#
#
#         """agent characteristics"""
#         self.id = unique_id
#         self.battery_capacity_n = 15                            # every household has an identical battery, for now
#         self.pv_generation = random.uniform(0, 1)               # random.choice(range(15)) * pvgeneration
#         self.consumption = random.uniform(0, 1)                 # random.choice(range(15)) * consumption
#         self.classification = []
#
#         """control variables"""
#         self.E_j_surplus = 0                                            # sellers
#         self.E_j_supply = 0
#         self.E_i_demand = 0                                             # buyers
#         self.E_i_allocation = 0                                         # buyers
#         self.payment_to_seller = 0                                      # buyer
#         self.w_j_storage_factor = 0
#         self.c_i_bidding_price = 0
#         self.stored = 0
#         self.bidding_prices_others = 0
#         self.E_supply_others = 0
#         self.w_nominal = 0
#
#         """ utilities of agents"""
#         self.utility_i = 0
#         self.utility_j = 0
#
#         """ merely a summary """
#         self.utilities_seller = [0, 0, 0]
#         self.utilities_buyer = [0, 0, 0, 0]
#
#         """prediction"""
#         self.current_step = 0
#         self.max_horizon =  700
#         self.horizon_agent = min(self.max_horizon, sim_steps - self.current_step)  # including current step
#         self.predicted_E_surplus_list = np.zeros(self.horizon_agent)
#         self.w_j_prediction = 0.5
#         """Battery related"""
#         self.soc_actual = 0 #self.battery_capacity_n
#         self.soc_preferred = self.soc_actual * 0.7
#         self.soc_gap = 0
#         self.soc_influx = 0
#         self.batt_available = self.battery_capacity_n - self.soc_actual
#
#         """ Actuator saturation of DER"""
#         self.actuator_sat_PV = get_PV_satuation(step_time)
#         self.actuator_sat_ESS_discharge, self.actuator_sat_ESS_charge = get_ESS_satuation(step_time)
#
#         self.soc_surplus = 0
#         self.charge_rate = 0
#         self.discharge_rate = 0
#         self.lower_bound_on_w_j = 0
#         self.deficit = 0
#         """results"""
#         self.results = []
#         self.tolerance_seller = 1
#         self.tolerance_buyer = 1
#         self.tol_buyer = []
#         self.action = []
#
#         self.E_prediction_agent = 0
#         self.w_prediction = 0
#         self.solar_capacity = 0
#
#         self.soc_actual = random.uniform(0.5 * self.soc_actual, self.battery_capacity_n)
#         self.solar_capacity = 1
#
#         self.revenue = 0
#         self.payment = 0
#
#         """Blockchain"""
#         if blockchain == 'on':
#             self.address_agent = w3.eth.accounts[addr]
#
#         self.promise_on_bc = 0
#         self.balance_on_bc = 0
#
#         self.profit = 0
#         self.E_j_actual_supplied = 0
#         self.E_j_returned_supply = 0
#
#         """ topology """
#         self.other_agents = agents_set
#         agent_numbers = range(0, N), range(0, N), range(0, N)
#         agent_numbers = np.reshape(agent_numbers, (N*3))
#
#         """ A number of agents are lagging a certain amounts of steps """
#         self.latency = random.randint()
#
#
#
#
#
#
#
#
#
#         """ topology"""
#         self.comm_reach = agent_numbers[:][(N + self.id - comm_radius) : (N + self.id + comm_radius + 1)]
#         self.c_bidding_prices_masked = np.zeros(2*comm_radius + 1)
#         self.E_total_supply_list_masked = np.zeros(2*comm_radius + 1)
#         self.E_total_supply = 0
#         self.R_total_masked = 0
#
#         self.R_total_list_masked = np.zeros(2*comm_radius + 1)
#         self.R_total = 0
#         self.E_supply_others_list_masked = np.zeros(2*comm_radius + 1)
#         self.E_supply_others_prediction_list_masked = np.zeros(2*comm_radius + 1)
#
#         self.E_supply_others_direct = 0
#         self.E_supply_others_prediction = 0
#
#
#     def step(self, big_data_file_per_step, big_data_file, E_total_surplus_prediction_per_step, horizon, prediction_range, steps, w3, contract_instance, agents_set, N):           # big_data_file = np.zeros((N, step_time, 3))
#         """Agent optimization step, what ever specific agents do on during step"""
#
#         self.E_j_actual_supplied = 0
#         self.E_j_returned_supply = 0
#
#         """real time data"""
#         self.consumption   = big_data_file_per_step[self.id, 0]           # import load file
#         self.pv_generation = big_data_file_per_step[self.id, 1]           # import generation file
#
#         """ agents personal prediction, can be different per agent (difference in prediction quality?)"""
#         self.horizon_agent = min(self.max_horizon, sim_steps - self.current_step - 1)  # including current step
#         self.predicted_E_surplus_list = np.zeros(self.horizon_agent)
#
#         for i in range(self.horizon_agent):
#             self.predicted_E_surplus_list[i] = big_data_file[steps + i][self.id][0] \
#                                                - big_data_file[steps + i][self.id][1]  # load - production
#             if self.predicted_E_surplus_list[i] < 0:
#                 self.predicted_E_surplus_list[i] = 0
#
#
#         self.w_prediction = calc_w_prediction() # has to go to agent
#         self.E_prediction_agent = calc_E_surplus_prediction(self.predicted_E_surplus_list,
#                                                             self.horizon_agent, N, prediction_range, steps) # from surplus
#
#         self.E_prediction_agent = self.E_prediction_agent * self.w_prediction # to actual supply
#
#         """Determine state of charge of agent's battery"""
#         self.current_step = steps
#         battery_horizon = self.horizon_agent  # including current step
#
#         self.soc_preferred = get_preferred_soc(self.soc_preferred, self.battery_capacity_n,
#                                                self.predicted_E_surplus_list, self.soc_actual, battery_horizon)
#         self.soc_gap = self.soc_preferred - self.soc_actual
#         self.soc_surplus = 0
#         if self.soc_gap < 0:
#             self.soc_surplus = abs(self.soc_gap)
#             self.soc_gap = 0
#
#
#         """determines in how many steps agent ideally wants to fill up its battery if possible"""
#         self.charge_rate = 10
#         self.discharge_rate = 10
#         self.lower_bound_on_w_j = 0
#
#
#         """define buyers/sellers classification"""
#         [classification, surplus_agent, demand_agent] = define_pool(self.consumption, self.pv_generation, self.soc_gap, self.soc_surplus, self.charge_rate, self.discharge_rate)
#         self.classification = classification
#
#         self.batt_available = self.battery_capacity_n - self.soc_actual
#
#         """Define players pool and let agents act according to battery states"""
#         if self.classification == 'buyer':
#             """buyers game init"""
#             self.E_i_demand = demand_agent
#             if self.soc_surplus >= self.E_i_demand:
#                 """ if buyer has battery charge left (after preferred_soc), it can either
#                 store this energy and do nothing, waiting until this surplus has gone,"""
#                 self.classification = 'passive'
#                 self.soc_actual -= demand_agent
#                 self.action = 'self-supplying from battery'
#                 # print("Household %d says: I am %s, %s" % (self.id, self.classification, self.action))
#             else:
#                 self.c_i_bidding_price = random.uniform(min(c_macro), max(c_macro))
#                 """values for sellers are set to zero"""
#                 self.action = 'bidding on the minutes-ahead market'
#                 # print("Household %d says: I am a %s" % (self.id, self.classification))
#             self.E_j_surplus = 0
#             self.w_j_storage_factor = 0
#
#         elif self.classification == 'seller':
#             """sellers game init"""
#             self.E_j_surplus = surplus_agent
#             if self.batt_available >= self.E_j_surplus:
#                 """agent can play as seller, since it needs available storage if w turns out to be 0"""
#                 self.w_j_storage_factor = random.uniform(0.2, 0.8)
#                 self.E_j_supply = calc_supply(surplus_agent, self.w_j_storage_factor)  # pool contains classification and E_j_surplus Ej per agents in this round
#                 self.lower_bound_on_w_j = 0
#                 # print("Household %d says: I am a %s" % (self.id, self.classification))
#                 self.action = 'selling to the grid'
#
#             elif self.batt_available < self.E_j_surplus:
#                 """the part that HAS to be sold otherwise the battery is filled up to its max"""
#                 self.lower_bound_on_w_j = (self.E_j_surplus - self.batt_available)/self.E_j_surplus
#                 self.w_j_storage_factor = random.uniform(self.lower_bound_on_w_j, 0.8)
#                 # print("Household %d says: I am a %s" % (self.id, self.classification))
#                 self.action = 'selling to the grid, (forced to due to overflowing battery)'
#
#             self.E_j_supply = self.E_j_surplus * self.w_j_storage_factor
#             """values for seller are set to zero"""
#             self.c_i_bidding_price = 0
#             self.E_i_demand = 0
#             self.payment_to_seller = 0
#
#         else:
#             self.c_i_bidding_price = 0
#             self.E_i_demand = 0
#             self.payment_to_seller = 0
#             self.E_j_supply = 0
#             self.E_j_surplus = 0
#             self.w_j_storage_factor = 0
#             # print("Household %d says: I am a %s agent" % (self.id, self.classification))
#
#         """ Broadcast promise through blockchain public ledger """
#         if blockchain == 'on':
#             self.broadcast_agent_info(w3, contract_instance)
#
#         """ MAKE TRANSACTION:
#             information on E_demand/E_surplus, initial c_i/w_j """
#
#
#     def optimization_buyers(self, N):
#         """preparation for update"""
#         pass
#
#
#     def optimization_seller(self, N):
#         pass
#
#
#     def broadcast_agent_info(self, w3, contract_instance, N):
#         if blockchain == 'off':
#             return
#
#         """ Setter on agent characteristics
#             Broadcast agents promise for this time-step"""
#         if self.classification == 'buyer':
#             c_i_broadcast = self.c_i_bidding_price
#             E_demand_broadcast = self.E_i_demand
#             self.w_j_storage_factor = 0
#             self.E_j_surplus = 0
#             self.promise_on_bc = setter_promise_buy(w3, contract_instance, self.address_agent, E_demand_broadcast, c_i_broadcast)
#         if self.classification == 'seller':
#             w_j_broadcast = self.w_j_storage_factor
#             E_surplus_broadcast = self.E_j_surplus
#             self.c_i_bidding_price = 0
#             self.E_i_demand = 0
#             self.promise_on_bc = setter_promise_sell(w3, contract_instance, self.address_agent, E_surplus_broadcast, w_j_broadcast)
#
#         # print('promise of agent %d = %d' % (self.id, self.promise_on_bc))
#
#         return
#         """
#         >> only integers in smart-contract... store promises in hash format?
# 		    NO: other agents recieve promises (needed information for opti) over blockchain, cannot be obscured?
# 		    YES: promise is only a requirement to trigger smart-contract, communication of those values are still communicated among peers off-chain.
# 	    Decide on this.."""
#
#     def settlement_of_payments(self, w3, contract_instance):
#         if blockchain == 'off':
#             return
#
#         """Listen to BC transactions on relevant promises of this agent
#            IFF promise == final offer then smart contract is triggered to pay"""
#         if self.classification == 'buyer':
#             # promise_on_bc = contract_instance.promiseOfbuy(self.address_agent)
#             # print(promise_on_bc, self.payment)
#             self.balance_on_bc = setter_burn(w3, contract_instance, self.address_agent, int(self.payment))
#             self.revenue = 0
#         if self.classification == 'seller':
#             # promise_on_bc = contract_instance.promiseOfsell(self.address_agent)
#             # print(promise_on_bc, self.payment)
#             self.balance_on_bc = setter_mint(w3, contract_instance, self.address_agent, self.revenue)
#             self.payment = 0
#
#
#         # print('balance of agent %d = %d' % (self.id, self.balance_on_bc))
#         return
#
#     def __repr__(self):
#         return "ID: %d, batterycapacity:%d, pvgeneration:%d, consumption:%d" % (self.id, self.battery_capacity, self.pv_generation, self.consumption)
#
#
# """Microgrid model environment"""
#
# class MicroGrid_async(Model):
#     print("Asynchronous Model")
#
#     """create environment in which agents can operate"""
#
#     def __init__(self, big_data_file, starting_point, N, comm_radius, lambda_set):
#
#         """Initialization and automatic creation of agents"""
#
#         self.steps = starting_point
#         self.time = starting_point
#
#         self.num_households = N
#         self.steps = 0
#         self.time = 0
#         self.big_data_file = big_data_file
#         self.agents = []
#         self.E_consumption_list  = np.zeros(N)
#         self.E_production_list  = np.zeros(N)
#         self.E_total_supply = 0
#         self.E_total_surplus = 0
#         self.c_bidding_prices = np.zeros(N)
#         self.c_nominal = 0
#         self.w_storage_factors = np.zeros(N)
#         self.w_nominal = 0
#         self.R_total = 0
#         self.R_list = np.zeros(N)
#         self.E_demand = 0
#         self.E_allocation_list = np.zeros(N)
#         self.E_total_supply_list = np.zeros(N)
#         self.E_total_demand_list = np.zeros(N)
#         self.E_allocation_total = 0
#         self.E_total_demand = 0
#         """Battery"""
#         self.E_total_stored = 0
#         self.E_total_surplus = 0
#         self.battery_soc_total = 0
#         self.actual_batteries = np.zeros(N)
#         """Unrelated"""
#         self.sellers_pool = []
#         self.buyers_pool = []
#         self.passive_pool = []
#         """Two pool prediction defining future load and supply"""
#         self.R_prediction = 0
#         self.E_prediction = 0
#         self.E_supply_prediction = 0
#         self.E_surplus_prediction_over_horizon = np.zeros(N)
#         self.w_prediction_avg = 0.5
#         self.c_nominal_prediction = 0
#         self.E_total_demand_prediction = 0
#         self.E_surplus_prediction_per_agent = 0
#         self.soc_preferred_list = np.zeros(N)
#
#         """ events"""
#         self.event_CreatedEnergy = classmethod
#         self.event_InitialisedContract = classmethod
#
#         """ prediction range is total amount of steps left """
#         self.prediction_range = sim_steps - self.steps  # data ends at end of day
#         if self.prediction_range <= 0:
#             self.prediction_range = 1
#
#         self.E_total_surplus_prediction_per_step = np.zeros(self.prediction_range)
#         self.utilities_sellers = np.zeros((N, 3))
#         self.utilities_buyers = np.zeros((N, 4))
#
#
#         """ compile and deploy smart-contract """
#         self.w3 = None
#         self.contract_instance = None
#         if blockchain == 'on':
#             contract_interface, self.w3 = compile_smart_contract()
#             creator_address = self.w3.eth.accounts[0]
#             self.w3, self.contract_instance, deployment_tx_hash, contract_address,\
#                                     self.event_CreatedEnergy, self.event_InitialisedContract = deploy_SC(contract_interface, self.w3, creator_address)
#             setter_initialise_tokens(self.w3, self.contract_instance, deployment_tx_hash, creator_address, self.event_InitialisedContract)
#
#         """create a set of N agents with activations schedule and e = unique id"""
#         for i in range(self.num_households):
#             addr = i + 1
#             agent = HouseholdAgent(i, self, self.w3, addr, self.agents, N, comm_radius)
#             self.agents.append(agent)
#
#         if blockchain == 'on':
#             for agent in self.agents[:]:
#                 setter_initialise_tokens2(self.w3, self.contract_instance, deployment_tx_hash, agent.address_agent)
#
#         """ settlement """
#         self.supply_deals = np.zeros(N)
#         self.E_surplus_list = np.zeros(N)
#         self.profit_list = np.zeros(N)
#
#         self.num_global_iteration = 0
#         self.num_buyer_iteration = 0
#         self.num_seller_iteration = 0
#
#         self.payed_list = np.zeros(N)
#         self.received_list = np.zeros(N)
#         self.deficit_total = 0
#         self.deficit_total_progress = 0
#         self.E_actual_supplied_list = np.zeros(N)
#         self.E_allocation_list = np.zeros(N)
#
#         self.revenue_list = np.zeros(N)
#         self.payment_list = np.zeros(N)
#
#         # """ Topology"""
#         # self.E_surplus_prediction_over_horizon_list = np.zeros(N)
#
#
#
#     def step(self, N, lambda_set):
#         """Environment proceeds a step after all agents took a step"""
#         print("Step =", self.steps)
#
#         if self.steps == 64:
#             print('220')
#             pass
#
#         for agent in self.agents[:]:
#             self.actual_batteries[agent.id] = agent.soc_actual
#
#         self.utilities_sellers = np.zeros((N, 3))
#         self.utilities_buyers = np.zeros((N, 4))
#
#         self.sellers_pool = []
#         self.buyers_pool = []
#         self.passive_pool = []
#
#         self.E_surplus_list = np.zeros(N)
#         self.E_total_supply_list = np.zeros(N)
#         self.E_total_demand_list = np.zeros(N)
#
#         self.E_total_surplus = 0
#         self.E_total_supply = 0
#         self.E_total_allocation = np.zeros(N)
#
#         self.num_global_iteration = 0
#         self.num_buyer_iteration = 0
#         self.num_seller_iteration = 0
#
#         """DYNAMICS"""
#         """ determine length/distance of horizon over which prediction data is effectual through a weight system (log, linear.. etc) """
#         horizon = min(70, sim_steps - self.steps)  # including current step
#
#         """ prediction range is total amount of steps left """
#         self.prediction_range = sim_steps - self.steps  # data ends at end of day
#         if self.prediction_range <= 0:
#             self.prediction_range = 1
#
#         """Take initial """
#         for agent in self.agents[:]:
#             agent.step(self.big_data_file[self.steps], self.big_data_file, self.E_total_surplus_prediction_per_step, horizon, self.prediction_range, self.steps, self.w3, self.contract_instance, self.agents, N)
#             if agent.classification == 'buyer':
#                 """Level 1 init game among buyers"""
#                 self.buyers_pool.append(agent.id)
#                 self.c_bidding_prices[agent.id] = agent.c_i_bidding_price
#                 self.E_total_demand_list[agent.id] = agent.E_i_demand
#                 agent.w_j_storage_factor = 0
#             elif agent.classification == 'seller':
#                 """Level 2 init of game of sellers"""
#                 self.sellers_pool.append(agent.id)
#                 self.w_storage_factors[agent.id] = agent.w_j_storage_factor
#                 self.E_total_surplus += agent.E_j_surplus                               # does not change by optimization
#                 self.E_surplus_list[agent.id] = agent.E_j_surplus
#                 agent.E_j_supply = agent.E_j_surplus * agent.w_j_storage_factor
#                 self.E_total_supply_list[agent.id] = agent.E_j_supply     # does change by optimization
#             else:
#                 self.passive_pool.append(agent.id)
#
#
#         if np.any(np.isnan(self.E_total_supply_list)):
#             print("some supply is NaN!?")
#             for agent in self.agents[:]:
#                 if np.isnan(self.E_total_supply_list[agent.id]):
#                     self.E_total_supply_list[agent.id] = 0
#
#
#         self.E_total_supply = sum(self.E_total_supply_list)
#         for agent in self.agents[:]:
#             agent.E_supply_others = sum(self.E_total_supply_list) - self.E_total_supply_list[agent.id]
#             self.E_allocation_list[agent.id] = agent.E_i_allocation
#
#         """Optimization after division of players into pools"""
#         tolerance_global = np.zeros(N)
#         tolerance_sellers = np.zeros(N)
#         tolerance_buyers = np.zeros(N)
#
#         iteration_global = 0
#         iteration_buyers = 0
#         iteration_sellers = 0
#         self.c_nominal = 0
#         self.R_total = 0  # resets for every step
#         self.R_prediction = 0
#
#         """Global optimization"""
#         initial_values = self.c_bidding_prices
#         payment_to_seller = np.zeros(N)
#
#         """global level"""
#         epsilon_buyers_list = []
#
#         while True: # global
#             iteration_global += 1
#             if len(self.buyers_pool) != 0 and len(self.sellers_pool) != 0:
#                 self.num_global_iteration += 1
#
#             prev_c_nominal = self.c_nominal
#             if self.E_total_supply == 0:
#                 for agent in self.agents[:]:
#                     agent.E_i_allocation = 0
#                     self.supply_deals[agent.id] = agent.E_i_allocation
#                 break
#             """Buyers level optimization"""
#             while True: # buyers
#                 iteration_buyers += 1
#                 if len(self.buyers_pool) != 0 and sum(self.E_surplus_list) != 0:
#                     self.num_buyer_iteration += 1
#                 """agent.c_i_bidding_price should incorporate its E_i_demand. If allocation is lower than the actual E_i_demand, payment to macro-grid
#                 (which will be more expensive than buying from Alice) needs to be minimised.
#                 Utility of buyer should be: (total energy demand - the part allocated from alice(c_i) ) * c_macro
#                 then allocation will be increased by offering more money"""
#
#                 for agent in self.agents[:]:
#                     if agent.classification == 'buyer':
#
#                         """ Build masks for asynchronous network or topology """
#                         for i in range(len(agent.comm_reach)):
#                                 agent.E_total_supply_list_masked[i] = self.E_total_supply_list[agent.comm_reach[i]]
#                                 agent.c_bidding_prices_masked[i] = self.c_bidding_prices[agent.comm_reach[i]]
#
#                         agent.E_total_supply = sum(agent.E_total_supply_list_masked)
#                         agent.bidding_prices_others = bidding_prices_others(agent.c_bidding_prices_masked, agent.c_i_bidding_price)
#                         self.E_allocation_list[agent.id] = allocation_i(agent.E_total_supply, agent.c_i_bidding_price, agent.bidding_prices_others)
#
#                         """preparation for update"""
#                         prev_bid = agent.c_i_bidding_price
#                         self.E_total_supply = sum(self.E_total_supply_list)
#                         agent.bidding_prices_others = bidding_prices_others(self.c_bidding_prices,
#                                                                             agent.c_i_bidding_price)
#
#                         """update c_i"""
#                         sol_buyer, sol_buyer.x[0] = buyers_game_optimization(agent.id,
#                                                                              agent.E_i_demand,
#                                                                              self.E_total_supply,
#                                                                              c_macro,
#                                                                              agent.c_i_bidding_price,
#                                                                              agent.bidding_prices_others,
#                                                                              agent.batt_available,
#                                                                              agent.soc_gap,
#                                                                              lambda_set, agent.actuator_sat_ESS_charge)
#
#
#                         if np.isnan(sol_buyer.x[0]):
#                             sol_buyer.x[0] = 0
#                             # exit("c_i is NaN")
#
#                         """ update values """
#                         agent.c_i_bidding_price = sol_buyer.x[0]
#                         self.c_bidding_prices[agent.id] = agent.c_i_bidding_price
#
#                         agent.bidding_prices_others = bidding_prices_others(agent.c_bidding_prices_masked, agent.c_i_bidding_price)
#                         agent.E_i_allocation = allocation_i(self.E_total_supply, agent.c_i_bidding_price,
#                                                             agent.bidding_prices_others)
#                         self.E_allocation_list[agent.id] = agent.E_i_allocation
#                         payment_to_seller[agent.id] = agent.c_i_bidding_price * agent.E_i_allocation
#                         agent.utility_i, demand_gap, utility_demand_gap, utility_costs = calc_utility_function_i(
#                             agent.E_i_demand,
#                             self.E_total_supply,
#                             agent.c_i_bidding_price,
#                             agent.bidding_prices_others,
#                             lambda_set)
#                         agent.utilities_buyer = [agent.utility_i, demand_gap, utility_demand_gap, utility_costs]
#                         self.utilities_buyers[agent.id] = [agent.utility_i, demand_gap, utility_demand_gap,
#                                                            utility_costs]
#
#                         if agent.utility_i - sol_buyer.fun > 1:
#                             # sys.exit("utility_i calculation does not match with optimization code")
#                             pass
#                         """ tolerances """
#                         new_bid = agent.c_i_bidding_price
#                         agent.tol_buyer.append(new_bid - prev_bid)
#                         tolerance_buyers[agent.id] = abs(new_bid - prev_bid)
#
#                         print('buyers trying to converge')
#                     else:
#                         self.c_bidding_prices[agent.id] = 0
#
#                     epsilon_buyers_game = max(abs(tolerance_buyers))
#                     epsilon_buyers_list.append(epsilon_buyers_game)
#
#                     """ END OF ROUND criteria """
#                     if epsilon_buyers_game < e_buyers:
#                         """ Values  to be plugged into sellers game"""
#                         if np.all(self.E_allocation_list == 0) or np.all(self.c_bidding_prices == 0):
#                             self.c_nominal = 0
#                         else:
#                             self.c_nominal = sum(self.E_allocation_list * self.c_bidding_prices) / sum(self.E_allocation_list)
#                         if sum(self.E_allocation_list) == 0:
#                             self.c_nominal = 0
#                         self.R_total = sum(payment_to_seller)
#                         for agent in self.agents[:]:
#                             self.E_allocation_total = agent.E_i_allocation
#                             agent.tol_buyer = []
#                         tolerance_buyers[:] = 0
#
#                         if np.any(self.E_allocation_list >= agent.actuator_sat_ESS_charge):
#                             # print("allocation is higher than charging actuator saturation")
#                             pass
#
#                         break
#                     else:
#                         pass
#
#             """ Determine global tolerances """
#             if sum(self.E_allocation_list) == 0:
#                 self.c_nominal = 0
#             if np.isnan(self.c_nominal):
#                 exit("c_i is NaN")
#                 pass
#             new_c_nominal = self.c_nominal
#             tolerance_c_nominal = abs(new_c_nominal - prev_c_nominal)
#
#             """sellers-level game optimization"""
#             while True: # sellers
#                 iteration_sellers += 1
#                 if len(self.sellers_pool) != 0 and sum(self.E_total_demand_list) != 0:
#                     self.num_seller_iteration += 1
#
#                 for agent in self.agents[:]:
#                     """ determine """
#                     agent.E_supply_others = sum(self.E_total_supply_list) - self.E_total_supply_list[agent.id]
#                     agent.bidding_prices_others = sum(self.c_bidding_prices) - agent.c_i_bidding_price
#                     agent.E_supply_others_prediction = self.E_supply_prediction - agent.E_supply_prediction_agent  #+ self.battery_soc_total - agent.battery_capacity_n
#
#                 for agent in self.agents[:]:
#                     if agent.classification == 'seller':
#
#                         """ Build masks for asynchronous network or topology """
#                         for i in range(len(agent.comm_reach)):
#                             agent.R_total_list_masked[i] = self.R_total_list[agent.comm_reach[i]] # done
#                         agent.R_total = self.R_total_list[agent.comm_reach].sum()
#                         agent.E_supply_others_direct = self.E_total_supply_list[agent.comm_reach].sum() - agent.E_j_supply
#                         agent.E_supply_others_prediction = (self.E_surplus_prediction_over_horizon_list[agent.comm_reach].sum() - agent.E_supply_prediction_agent) * self.w_prediction_avg
#
#                         """ Sellers optimization game, plugging in bidding price to decide on sharing factor.
#                                                 Is bidding price lower than that the smart-meter expects to sell on a later time period?
#                                                 smart-meter needs a prediction on the coming day. Either use the load data or make a predicted model on
#                                                 all aggregated load data """
#                         prev_wj = agent.w_j_storage_factor
#                         agent.bidding_prices_others = sum(self.c_bidding_prices)
#                         """ Optimization """
#
#                         if prediction == 'on':
#                             sol_seller, \
#                             sol_seller.x[0], \
#                             utility_seller_function = sellers_game_optimization(agent.id,
#                                                                                 agent.E_j_surplus,
#                                                                                 agent.R_total,
#                                                                                 agent.E_supply_others,
#                                                                                 agent.R_prediction,
#                                                                                 agent.E_supply_others_prediction,
#                                                                                 agent.w_j_storage_factor,
#                                                                                 agent.E_supply_prediction_agent,
#                                                                                 agent.lower_bound_on_w_j,
#                                                                                 lambda_set,
#                                                                                 agent.actuator_sat_ESS_discharge)
#
#                             agent.w_j_storage_factor = sol_seller.x[0]
#                             prediction_utility, direct_utility, agent.utility_j = calc_utility_function_j(agent.id,
#                                                                                                           agent.E_j_surplus,
#                                                                                                           agent.R_total,
#                                                                                                           agent.E_supply_others,
#                                                                                                           agent.R_prediction,
#                                                                                                           agent.E_supply_others_prediction,
#                                                                                                           agent.w_j_storage_factor,
#                                                                                                           agent.E_supply_prediction_agent,
#                                                                                                           lambda_set)
#
#                             if abs(agent.utility_j - sol_seller.fun) > 10:
#                                 # exit("utility_j calculation does not match with optimization code")
#                                 pass
#
#                             agent.utility_seller = [agent.utility_j, prediction_utility, direct_utility]
#
#                         if prediction == 'off':
#                             sol_seller, \
#                             sol_seller.x[0], \
#                             utility_seller_function = sellers_game_optimization(agent.id,
#                                                                                 agent.E_j_surplus,
#                                                                                 agent.R_total,
#                                                                                 agent.E_supply_others,
#                                                                                 agent.R_prediction,
#                                                                                 agent.E_supply_others_prediction,
#                                                                                 agent.w_j_storage_factor,
#                                                                                 agent.E_supply_prediction_agent,
#                                                                                 agent.lower_bound_on_w_j,
#                                                                                 lambda_set,
#                                                                                 agent.actuator_sat_ESS_discharge)
#
#                             agent.utility_seller = [utility_seller_function, None, None]
#                         self.utilities_sellers[agent.id] = agent.utility_seller
#
#                         """"""""""""""""""""""""""""""
#                         """ In case of ALL-supply """
#                         """"""""""""""""""""""""""""""
#                         # agent.w_j_storage_factor = 1
#                         # agent.E_j_supply = agent.E_j_surplus
#
#                         """ Update on values """
#                         agent.E_j_supply = agent.E_j_surplus * agent.w_j_storage_factor
#                         self.E_total_supply_list[agent.id] = agent.E_j_supply
#                         if self.E_total_supply_list[agent.id] <= 0 or np.isnan(self.E_total_supply_list[agent.id]):
#                             self.E_total_supply_list[agent.id] = 0
#
#                         self.E_total_supply = sum(self.E_total_supply_list)
#
#                         agent.E_supply_others = sum(self.E_total_supply_list) - self.E_total_supply_list[agent.id]
#                         self.w_storage_factors[agent.id] = agent.w_j_storage_factor
#
#                         """Tolerance"""
#                         new_wj = agent.w_j_storage_factor
#                         tolerance_sellers[agent.id] = abs(new_wj - prev_wj)
#                     else:
#                         agent.w_j_storage_factor = 0
#
#                     self.E_total_supply = sum(self.E_total_supply_list)
#                     self.E_demand = sum(self.E_total_demand_list)
#
#                     for agent in self.agents[:]:
#                         agent.E_supply_others = sum(self.E_total_supply_list) - self.E_total_supply_list[agent.id]
#
#                     self.w_nominal = sum(self.E_total_supply_list) / self.E_total_surplus
#                     epsilon_sellers_game = max(abs(tolerance_sellers))
#
#                     """ END OF ROUND criteria """
#                     if epsilon_sellers_game < e_sellers:
#                         tolerance_sellers[:] = 0
#                         self.E_total_supply = sum(self.E_total_supply_list)
#                         supply_new = self.E_total_supply
#                         """ Update on batteries """
#                         for agent in self.agents[:]:
#                             self.actual_batteries[agent.id] = agent.soc_actual
#
#                         if np.any(self.E_total_supply_list >= agent.actuator_sat_ESS_discharge):
#                             # print("supply is higher than actuator saturation")
#                             pass
#
#                         break
#                     else:
#                         pass
#
#                 tolerance_supply = abs(supply_new - supply_old)
#                 epsilon_c_nominal = abs(tolerance_c_nominal)
#                 self.E_actual_supplied_list = np.zeros(N)
#                 """ END OF ROUND criteria """
#                 if (epsilon_c_nominal < e_cn and tolerance_supply < e_supply) or self.num_global_iteration > 200:
#                     for agent in self.agents[:]:
#                         self.E_allocation_list[agent.id] = agent.E_i_allocation
#                     self.E_total_supply = sum(self.E_total_supply_list)
#                     self.E_allocation_total = sum(self.E_allocation_list)
#                     ratio_oversupply = self.E_allocation_total / self.E_total_supply
#
#                     if ratio_oversupply > 1.0:
#                         """ Sellers are offering more energy than demand on the market, profit is divided """
#                         for agent in self.agents[:]:
#                             agent.E_j_actual_supplied = agent.E_j_supply / ratio_oversupply
#                             agent.E_j_returned_supply = agent.E_j_supply - agent.E_j_actual_supplied
#                             self.E_actual_supplied_list[agent.id] = agent.E_j_actual_supplied
#                     elif ratio_oversupply <= 1.0:
#                         """ Sellers are offering less energy than demand on the market """
#                         for agent in self.agents[:]:
#                             agent.E_j_actual_supplied = agent.E_j_supply
#                             agent.E_j_returned_supply = 0
#                             self.E_actual_supplied_list[agent.id] = agent.E_j_actual_supplied
#                     break
#                 else:
#                     pass
#
#         self.deficit_total = 0
#         self.supply_deals = np.zeros(N)
#         total_payment = 0
#
#         """settle all deals"""
#         for agent in self.agents[:]:
#             agent.deficit = 0
#             agent.payment = 0
#             agent.soc_influx = 0
#             if agent.classification == 'buyer':
#                 """ buyers costs """
#                 self.supply_deals[agent.id] = agent.E_i_allocation
#                 agent.soc_influx = agent.E_i_allocation - agent.consumption + agent.pv_generation
#                 if agent.soc_actual + agent.soc_influx < 0:
#                     """ battery is depleting """
#                     agent.deficit = abs(agent.soc_actual + agent.soc_influx)
#                     agent.soc_actual = 0
#                 elif agent.soc_actual + agent.soc_influx > agent.battery_capacity_n:
#                     """ battery is overflowing """
#                     real_influx = agent.soc_influx
#                     agent.soc_influx = agent.battery_capacity_n - agent.soc_actual
#                     agent.batt_overflow = real_influx - agent.soc_influx
#                 else:
#                     agent.soc_actual += agent.soc_influx
#                 agent.payment = agent.E_i_allocation * agent.c_i_bidding_price
#                 total_payment += agent.payment
#             self.actual_batteries[agent.id] = agent.soc_actual
#
#             agent.revenue = 0
#             if agent.classification == 'seller':
#                 """ sellers earnings """
#                 agent.soc_influx = agent.E_j_surplus * (1 - agent.w_j_storage_factor) + agent.E_j_returned_supply
#                 if agent.soc_actual + agent.soc_influx < 0:
#                     """ battery is depleting """
#                     agent.deficit = abs(agent.soc_actual + agent.soc_influx)
#                     agent.soc_influx = agent.E_i_demand + agent.deficit
#                 if agent.soc_actual + agent.soc_influx > agent.battery_capacity_n:
#                     """ battery is overflowing """
#                     real_influx = agent.soc_influx
#                     agent.soc_influx = agent.battery_capacity_n - agent.soc_actual
#                     agent.batt_overflow = real_influx - agent.soc_influx
#                 agent.revenue = agent.E_j_actual_supplied / sum(self.E_actual_supplied_list) * total_payment
#                 if np.isnan(agent.revenue):
#                     agent.revenue = 0
#                 agent.soc_actual += agent.soc_influx
#
#                 """ HERE W IS UPDATED WRT OVERSUPPLY!"""
#                 agent.w_j_storage_factor = agent.soc_influx / agent.E_j_surplus
#
#             self.deficit_total += abs(agent.deficit)
#
#             self.actual_batteries[agent.id] = agent.soc_actual
#         if np.any(self.actual_batteries < 0) or np.any(self.actual_batteries > agent.battery_capacity_n):
#             exit("negative battery soc, physics are broken")
#
#         if np.any(self.actual_batteries) < 0:
#             print("negative battery soc")
#
#         self.deficit_total_progress += self.deficit_total
#         self.battery_soc_total = sum(self.actual_batteries)
#         if self.battery_soc_total < 0:
#             print("negative battery soc")
#
#         """ Pull data out of agents """
#         total_soc_pref = 0
#         for agent in self.agents[:]:
#             total_soc_pref += agent.soc_preferred
#             self.soc_preferred_list[agent.id] = agent.soc_preferred
#         avg_soc_preferred = total_soc_pref / N
#
#         """ Blockchain """
#         if blockchain == 'on':
#             for agent in self.agents[:]:
#                 agent.settlement_of_payments(self.w3, self.contract_instance, self.steps)
#
#         self.profit_list = np.zeros(N)
#         """ Costs on this step"""
#         self.profit_list = np.zeros(N)
#         self.revenue_list = np.zeros(N)
#         self.payed_list = np.zeros(N)
#         for agent in self.agents[:]:
#             agent.profit = agent.revenue - agent.payment
#             self.profit_list[agent.id] = agent.profit
#             self.revenue_list[agent.id] = agent.revenue
#             self.payment_list[agent.id] = agent.payment
#
#         """ Update time """
#         self.steps += 1
#         self.time += 1
#
#         return self.E_total_surplus, self.E_demand, \
#                self.buyers_pool, self.sellers_pool, self.w_storage_factors, \
#                self.c_nominal, self.w_nominal, \
#                self.R_prediction, self.E_supply_prediction, self.R_total, \
#                self.actual_batteries, self.E_total_supply, \
#                self.utilities_buyers, self.utilities_sellers, \
#                self.soc_preferred_list, avg_soc_preferred, \
#                self.E_consumption_list, self.E_production_list, \
#                self.E_total_demand_list, self.c_bidding_prices, self.E_surplus_list, self.E_total_supply_list, \
#                self.num_global_iteration, self.num_buyer_iteration, self.num_seller_iteration, \
#                self.profit_list, self.revenue_list, self.payment_list, \
#                self.deficit_total, self.deficit_total_progress, self.E_actual_supplied_list, self.E_allocation_list
#
#
#     def __repr__(self):
#         return "synchronous Microgrid"
#
