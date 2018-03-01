from blockchain.smartcontract import *
from source.function_file import *
import sys

import numpy as np
from mesa import Agent, Model


##########################
### ASYNCHRONOUS MODEL ###
##########################
""" noise: 0 is the mean of the normal distribution you are choosing from
           1 is the standard deviation of the normal distribution
           100 is the number of elements you get in array noise    """
noise = np.random.normal(0, 1, 100)


"""All starting parameters are initialised"""
starting_point = 0
stopping_point = 7200 - starting_point - 2000
step_day = 1440
days = 5
blockchain = 'off'

step_time = 100
total_steps = step_day*days
sim_steps = int(total_steps/step_time)

step_list = np.zeros([sim_steps])

c_S = 10                                             # c_S is selling price of the microgrid
c_B = 1                                              # c_B is buying price of the microgrid
c_macro = (c_B, c_S)                                 # Domain of available prices for bidding player i
possible_c_i = range(c_macro[0], c_macro[1])         # domain of solution for bidding price c_i

e_buyers = 0.001
e_sellers = 0.001
e_global = 0.01
e_cn = 0.01
e_supply = 0.1

class HouseholdAgent(Agent):

    """All microgrid household(agents) should be generated here; initialisation of prosumer tools """
    def __init__(self, unique_id, model, w3, addr, agents_set, N, comm_radius):
        super().__init__(unique_id, model)

        """agent characteristics"""
        self.id = unique_id
        self.battery_capacity_n = 3000                          # every household has an identical battery, for now
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
        self.E_supply_prediction_agent = 0

        """Battery related"""
        self.soc_preferred = 1000
        self.soc_actual = 1000
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

        self.E_surplus_prediction_agent = 0
        self.w_prediction = 0
        self.solar_capacity = 0

        # agent_char = np.ones(N)
        self.soc_actual = random.uniform(0.5 * self.soc_actual, 0.8 * self.soc_actual)
        self.solar_capacity = 1

        self.revenue = 0
        self.payment = 0

        """Blockchain"""
        if blockchain == 'on':
            self.address_agent = w3.eth.accounts[addr]

        self.promise_on_bc = 0
        self.balance_on_bc = 0

        """ topology """
        self.other_agents = agents_set
        agent_numbers = range(0, N), range(0, N),range(0, N)
        agent_numbers = np.reshape(agent_numbers, (N*3))

        self.comm_reach = agent_numbers[:][(N + self.id - comm_radius) : (N + self.id + comm_radius + 1)]
        self.c_bidding_prices_masked = np.zeros(2*comm_radius + 1)
        self.E_total_supply_list_masked = np.zeros(2*comm_radius + 1)
        self.E_total_supply = 0
        self.R_total_masked = 0

        self.R_total_list_masked = np.zeros(2*comm_radius + 1)
        self.R_total = 0
        self.E_supply_others_list_masked = np.zeros(2*comm_radius + 1)
        self.E_supply_others_prediction_list_masked = np.zeros(2*comm_radius + 1)

        self.E_supply_others_direct = 0
        self.E_supply_others_prediction = 0


    def step(self, big_data_file_per_step, big_data_file, E_total_surplus_prediction_per_step, horizon, prediction_range, steps, w3, contract_instance, agents_set, N):           # big_data_file = np.zeros((N, step_time, 3))
        """Agent optimization step, what ever specific agents do on during step"""

        """real time data"""
        self.consumption   = big_data_file_per_step[self.id, 0]           # import load file
        self.pv_generation = big_data_file_per_step[self.id, 1] * self.solar_capacity        # import generation file

        """ agents personal prediction, can be different per agent (difference in prediction quality?)"""
        self.horizon_agent = min(self.max_horizon, sim_steps - self.current_step - 1)  # including current step
        self.predicted_E_surplus_list = np.zeros(self.horizon_agent)

        for i in range(self.horizon_agent):
            self.predicted_E_surplus_list[i] = big_data_file[steps + i][self.id][0] \
                                               - big_data_file[steps + i][self.id][1]  # load - production
            if self.predicted_E_surplus_list[i] < 0:
                self.predicted_E_surplus_list[i] = 0

        # self.E_n_surplus_prediction_one = big_data_file[steps][self.id][0] - big_data_file[steps][self.id][1]

        self.w_prediction = calc_w_prediction() # has to go to agent
        self.E_surplus_prediction_agent = calc_E_surplus_prediction(self.predicted_E_surplus_list,
                                                            self.horizon_agent, N, prediction_range, steps) # from surplus

        self.E_supply_prediction_agent = self.E_surplus_prediction_agent * self.w_prediction # to actual supply

        """Determine state of charge of agent's battery"""
        self.current_step = steps
        battery_horizon = self.horizon_agent  # including current step

        self.soc_preferred = get_preferred_soc(self.soc_preferred, self.battery_capacity_n,
                                               self.predicted_E_surplus_list, battery_horizon)
        soc_gap = self.soc_preferred - self.soc_actual
        self.soc_gap = soc_gap
        self.soc_surplus = 0
        if self.soc_gap < 0:
            self.soc_surplus = abs(soc_gap)
            self.soc_gap = 0


        """determines in how many steps agent ideally wants to fill up its battery if possible"""
        self.charge_rate = 10
        self.discharge_rate = 10
        self.lower_bound_on_w_j = 0

        """define buyers/sellers classification"""
        [classification, surplus_agent, demand_agent] = define_pool(self.consumption, self.pv_generation, self.soc_gap, self.soc_surplus, self.charge_rate, self.discharge_rate)
        self.classification = classification

        self.batt_available = self.battery_capacity_n - self.soc_actual

        """Define players pool and let agents act according to battery states"""
        if self.classification == 'buyer':
            """buyers game init"""
            self.E_i_demand = demand_agent
            if self.soc_surplus >= self.E_i_demand:
                """ if buyer has battery charge left (after preferred_soc), it can either 
                store this energy and do nothing, waiting until this surplus has gone,"""
                self.classification = 'passive'
                self.soc_actual -= demand_agent
                self.action = 'self-supplying from battery'
                # print("Household %d says: I am %s, %s" % (self.id, self.classification, self.action))
            else:
                self.c_i_bidding_price = random.uniform(min(c_macro), max(c_macro))
                """values for sellers are set to zero"""
                self.action = 'bidding on the minutes-ahead market'
                # print("Household %d says: I am a %s" % (self.id, self.classification))
            self.E_j_surplus = 0
            self.w_j_storage_factor = 0

        elif self.classification == 'seller':
            """sellers game init"""
            self.E_j_surplus = surplus_agent
            if self.batt_available >= self.E_j_surplus:
                """agent can play as seller, since it needs available storage if w turns out to be 0"""
                self.w_j_storage_factor = random.uniform(0.2, 0.8)
                self.E_j_supply = calc_supply(surplus_agent, self.w_j_storage_factor)  # pool contains classification and E_j_surplus Ej per agents in this round
                self.lower_bound_on_w_j = 0
                # print("Household %d says: I am a %s" % (self.id, self.classification))
                self.action = 'selling to the grid'

            elif self.batt_available < self.E_j_surplus:
                """the part that HAS to be sold otherwise the battery is filled up to its max"""
                self.lower_bound_on_w_j = (self.E_j_surplus - self.batt_available)/self.E_j_surplus
                self.w_j_storage_factor = random.uniform(self.lower_bound_on_w_j, 0.8)
                # print("Household %d says: I am a %s" % (self.id, self.classification))
                self.action = 'selling to the grid, (forced to due to overflowing battery)'

            self.E_j_supply = self.E_j_surplus * self.w_j_storage_factor
            """values for seller are set to zero"""
            self.c_i_bidding_price = 0
            self.E_i_demand = 0
            self.payment_to_seller = 0

        else:
            self.c_i_bidding_price = 0
            self.E_i_demand = 0
            self.payment_to_seller = 0
            self.E_j_supply = 0
            self.E_j_surplus = 0
            self.w_j_storage_factor = 0
            # print("Household %d says: I am a %s agent" % (self.id, self.classification))

        """ Broadcast promise through blockchain public ledger """
        if blockchain == 'on':
            self.broadcast_agent_info(w3, contract_instance)

        """ MAKE TRANSACTION:
            information on E_demand/E_surplus, initial c_i/w_j """


    def optimization_buyers(self, N):
        """preparation for update"""
        pass


    def optimization_seller(self, N):
        pass


    def broadcast_agent_info(self, w3, contract_instance, N):
        if blockchain == 'off':
            return

        """ Setter on agent characteristics
            Broadcast agents promise for this time-step"""
        if self.classification == 'buyer':
            c_i_broadcast = self.c_i_bidding_price
            E_demand_broadcast = self.E_i_demand
            self.w_j_storage_factor = 0
            self.E_j_surplus = 0
            self.promise_on_bc = setter_promise_buy(w3, contract_instance, self.address_agent, E_demand_broadcast, c_i_broadcast)
        if self.classification == 'seller':
            w_j_broadcast = self.w_j_storage_factor
            E_surplus_broadcast = self.E_j_surplus
            self.c_i_bidding_price = 0
            self.E_i_demand = 0
            self.promise_on_bc = setter_promise_sell(w3, contract_instance, self.address_agent, E_surplus_broadcast, w_j_broadcast)

        # print('promise of agent %d = %d' % (self.id, self.promise_on_bc))

        return
        """ 
        >> only integers in smart-contract... store promises in hash format? 
		    NO: other agents recieve promises (needed information for opti) over blockchain, cannot be obscured?
		    YES: promise is only a requirement to trigger smart-contract, communication of those values are still communicated among peers off-chain.
	    Decide on this.."""

    def settlement_of_payments(self, w3, contract_instance):
        if blockchain == 'off':
            return

        """Listen to BC transactions on relevant promises of this agent
           IFF promise == final offer then smart contract is triggered to pay"""
        if self.classification == 'buyer':
            # promise_on_bc = contract_instance.promiseOfbuy(self.address_agent)
            # print(promise_on_bc, self.payment)
            self.balance_on_bc = setter_burn(w3, contract_instance, self.address_agent, int(self.payment))
            self.revenue = 0
        if self.classification == 'seller':
            # promise_on_bc = contract_instance.promiseOfsell(self.address_agent)
            # print(promise_on_bc, self.payment)
            self.balance_on_bc = setter_mint(w3, contract_instance, self.address_agent, self.revenue)
            self.payment = 0


        # print('balance of agent %d = %d' % (self.id, self.balance_on_bc))
        return

    def __repr__(self):
        return "ID: %d, batterycapacity:%d, pvgeneration:%d, consumption:%d" % (self.id, self.battery_capacity, self.pv_generation, self.consumption)


"""Microgrid model environment"""

class MicroGrid_async(Model):
    print("Asynchronous Model")

    """create environment in which agents can operate"""

    def __init__(self, big_data_file, starting_point, N, comm_radius, lambda_set):

        """Initialization and automatic creation of agents"""

        self.steps = starting_point
        self.time = starting_point

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
        self.w_nominal = 0
        self.R_total = 0
        self.R_total_list = np.zeros(N)
        self.E_supply_per_agent = np.zeros(N)
        self.E_demand = 0
        self.E_allocation_per_agent = np.zeros(N)
        self.E_total_supply_list = np.zeros(N)
        self.E_total_demand_list = np.zeros(N)

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
        self.E_surplus_prediction_over_horizon = np.zeros(N)
        self.w_prediction_avg = 0.5
        self.c_nominal_prediction = 0
        self.E_total_demand_prediction = 0
        self.E_surplus_prediction_per_agent = 0

        """ events"""
        self.event_CreatedEnergy = classmethod
        self.event_InitialisedContract = classmethod

        """ prediction range is total amount of steps left """
        self.prediction_range = sim_steps - self.steps  # data ends at end of day
        if self.prediction_range <= 0:
            self.prediction_range = 1

        self.E_total_surplus_prediction_per_step = np.zeros(self.prediction_range)
        self.utilities_sellers = np.zeros((N, 3))
        self.utilities_buyers = np.zeros((N, 4))

        self.soc_preferred = np.zeros(N)

        """ compile and deploy smart-contract """
        self.w3 = None
        self.contract_instance = None
        if blockchain == 'on':
            contract_interface, self.w3 = compile_smart_contract()
            creator_address = self.w3.eth.accounts[0]
            self.w3, self.contract_instance, deployment_tx_hash, contract_address,\
                                    self.event_CreatedEnergy, self.event_InitialisedContract = deploy_SC(contract_interface, self.w3, creator_address)
            setter_initialise_tokens(self.w3, self.contract_instance, deployment_tx_hash, creator_address, self.event_InitialisedContract)

        """create a set of N agents with activations schedule and e = unique id"""
        for i in range(self.num_households):
            addr = i + 1
            agent = HouseholdAgent(i, self, self.w3, addr, self.agents, N, comm_radius)
            self.agents.append(agent)

        if blockchain == 'on':
            for agent in self.agents[:]:
                setter_initialise_tokens2(self.w3, self.contract_instance, deployment_tx_hash, agent.address_agent)


        """ Topology"""
        self.E_surplus_prediction_over_horizon_list = np.zeros(N)
        """ Deals """
        self.supply_deals = np.zeros(N)
        self.E_surplus_list = np.zeros(N)
        self.E_total_allocation = np.zeros(N)

        self.num_global_iteration = 0
        self.num_buyer_iteration = 0
        self.num_seller_iteration = 0

        self.deficit_total = 0
        self.profit_list = np.zeros(N)
        self.payed_list = np.zeros(N)
        self.received_list = np.zeros(N)

    def step(self, N, lambda_set):
        """Environment proceeds a step after all agents took a step"""
        print("Step =", self.steps)

        if self.steps == 64:
            print('220')
            pass

        for agent in self.agents[:]:
            self.actual_batteries[agent.id] = agent.soc_actual

        self.utilities_sellers = np.zeros((N, 3))
        self.utilities_buyers = np.zeros((N, 4))

        self.sellers_pool = []
        self.buyers_pool = []
        self.passive_pool = []

        self.E_surplus_list = np.zeros(N)
        self.E_total_supply_list = np.zeros(N)
        self.E_total_demand_list = np.zeros(N)

        self.E_total_surplus = 0
        self.E_total_supply = 0
        self.E_total_allocation = np.zeros(N)

        self.num_global_iteration = 0
        self.num_buyer_iteration = 0
        self.num_seller_iteration = 0

        """DYNAMICS"""
        """ determine length/distance of horizon over which prediction data is effectual through a weight system (log, linear.. etc) """
        horizon = min(70, sim_steps - self.steps)  # including current step

        """ prediction range is total amount of steps left """
        self.prediction_range = sim_steps - self.steps  # data ends at end of day
        if self.prediction_range <= 0:
            self.prediction_range = 1

        """Take initial """
        for agent in self.agents[:]:
            agent.step(self.big_data_file[self.steps], self.big_data_file, self.E_total_surplus_prediction_per_step, horizon, self.prediction_range, self.steps, self.w3, self.contract_instance, self.agents, N)
            if agent.classification == 'buyer':
                """Level 1 init game among buyers"""
                self.buyers_pool.append(agent.id)
                self.c_bidding_prices[agent.id] = agent.c_i_bidding_price
                self.E_total_demand_list[agent.id] = agent.E_i_demand
                agent.w_j_storage_factor = 0
            elif agent.classification == 'seller':
                """Level 2 init of game of sellers"""
                self.sellers_pool.append(agent.id)
                self.w_storage_factors[agent.id] = agent.w_j_storage_factor
                self.E_total_surplus += agent.E_j_surplus                               # does not change by optimization
                self.E_surplus_list[agent.id] = agent.E_j_surplus
                agent.E_j_supply = agent.E_j_surplus * agent.w_j_storage_factor
                self.E_total_supply_list[agent.id] = agent.E_j_supply     # does change by optimization
            else:
                self.passive_pool.append(agent.id)


        if np.any(np.isnan(self.E_total_supply_list)):
            print("some supply is NaN!?")
            for agent in self.agents[:]:
                if np.isnan(self.E_total_supply_list[agent.id]):
                    self.E_total_supply_list[agent.id] = 0


        self.E_total_supply = sum(self.E_total_supply_list)
        for agent in self.agents[:]:
            agent.E_supply_others = sum(self.E_total_supply_list) - self.E_total_supply_list[agent.id]
            self.E_allocation_per_agent[agent.id] = agent.E_i_allocation

        """Optimization after division of players into pools"""
        tolerance_global = np.zeros(N)
        tolerance_sellers = np.zeros(N)
        tolerance_buyers = np.zeros(N)

        iteration_global = 0
        iteration_buyers = 0
        iteration_sellers = 0
        self.c_nominal = 0
        self.R_total = 0  # resets for every step
        self.R_prediction = 0

        """Global optimization"""
        initial_values = self.c_bidding_prices
        payment_to_seller = np.zeros(N)

        """global level"""
        epsilon_buyers_list = []

        while True: # global
            iteration_global += 1
            if len(self.buyers_pool) != 0 and len(self.sellers_pool) != 0:
                self.num_global_iteration += 1

            prev_c_nominal = self.c_nominal
            if self.E_total_supply == 0:
                for agent in self.agents[:]:
                    agent.E_i_allocation = 0
                    self.supply_deals[agent.id] = agent.E_i_allocation
                break
            """Buyers level optimization"""
            while True: # buyers
                iteration_buyers += 1
                if len(self.buyers_pool) != 0 and sum(self.E_surplus_list) != 0:
                    self.num_buyer_iteration += 1
                """agent.c_i_bidding_price should incorporate its E_i_demand. If allocation is lower than the actual E_i_demand, payment to macro-grid 
                (which will be more expensive than buying from Alice) needs to be minimised. 
                Utility of buyer should be: (total energy demand - the part allocated from alice(c_i) ) * c_macro
                then allocation will be increased by offering more money"""

                for agent in self.agents[:]:
                    if agent.classification == 'buyer':
                        """preparation for update"""
                        prev_bid = agent.c_i_bidding_price
                        """ Build masks """
                        for i in range(len(agent.comm_reach)):
                                agent.E_total_supply_list_masked[i] = self.E_total_supply_list[agent.comm_reach[i]]
                                agent.c_bidding_prices_masked[i] = self.c_bidding_prices[agent.comm_reach[i]]

                        agent.E_total_supply = sum(agent.E_total_supply_list_masked)
                        agent.bidding_prices_others = bidding_prices_others(agent.c_bidding_prices_masked, agent.c_i_bidding_price)
                        self.E_allocation_per_agent[agent.id] = allocation_i(agent.E_total_supply, agent.c_i_bidding_price, agent.bidding_prices_others)

                        """update c_i using local information, agent.comm_reach functions as a mask """
                        sol_buyer, sol_buyer.x[0] = buyers_game_optimization(agent.id, agent.E_i_demand, agent.E_total_supply, c_macro, agent.c_i_bidding_price, agent.bidding_prices_others, agent.batt_available, agent.soc_gap, lambda_set)
                        agent.c_i_bidding_price =  sol_buyer.x[0]

                        if np.isnan(agent.c_i_bidding_price):
                            # exit("c_i is NaN")
                            agent.c_i_bidding_price = 0
                            agent.classification = 'passive'
                            tolerance_buyers[agent.id] = 0
                            print("agent %d is set to passive" % agent.id)

                        self.c_bidding_prices[agent.id] = agent.c_i_bidding_price
                        agent.bidding_prices_others = bidding_prices_others(agent.c_bidding_prices_masked, agent.c_i_bidding_price)
                        self.E_allocation_per_agent[agent.id] = allocation_i(agent.E_total_supply, agent.c_i_bidding_price, agent.bidding_prices_others)
                        agent.E_i_allocation = self.E_allocation_per_agent[agent.id]
                        payment_to_seller[agent.id] = agent.c_i_bidding_price * agent.E_i_allocation

                        agent.utility_i, demand_gap, utility_demand_gap, utility_costs = calc_utility_function_i(agent.E_i_demand, agent.E_total_supply, agent.c_i_bidding_price, agent.bidding_prices_others, lambda_set)
                        agent.utilities_buyer = [agent.utility_i, demand_gap, utility_demand_gap, utility_costs]
                        self.utilities_buyers[agent.id] = [agent.utility_i, demand_gap, utility_demand_gap, utility_costs]
                        self.R_total_list[agent.id] = self.E_allocation_per_agent[agent.id] * self.c_bidding_prices[agent.id]


                        # if agent.utility_i - sol_buyer.fun > 10:
                        #     sys.exit("utility_i calculation does not match with optimization code")
                        #     pass

                        new_bid = agent.c_i_bidding_price
                        agent.tol_buyer.append(new_bid - prev_bid)
                        tolerance_buyers[agent.id] = abs(new_bid - prev_bid)
                    else:
                        self.c_bidding_prices[agent.id] = 0  # current total payment offered

                epsilon_buyers_game = max(abs(tolerance_buyers))
                epsilon_buyers_list.append(epsilon_buyers_game)
                if epsilon_buyers_game < e_buyers:
                    """ Values to be plugged into sellers game"""
                    self.c_nominal = sum(self.R_total_list) / sum(self.E_allocation_per_agent)
                    if sum(self.E_allocation_per_agent) == 0:
                        self.c_nominal = 0
                    self.R_total = sum(payment_to_seller)
                    for agent in self.agents[:]:
                        self.R_total_list[agent.id] = self.E_allocation_per_agent[agent.id] * self.c_bidding_prices[agent.id]
                    tolerance_buyers[:] = 0
                    for agent in self.agents[:]:
                        agent.tol_buyer = []
                    break
                else:
                    pass

            """Determine global tolerances"""
            self.c_nominal = sum(self.E_allocation_per_agent * self.c_bidding_prices) / sum(self.E_allocation_per_agent)
            if np.isnan(self.c_nominal):
                exit("c_n is NaN")
                pass
            new_c_nominal = self.c_nominal
            tolerance_c_nominal = abs(new_c_nominal - prev_c_nominal)

            """ surplus_prediction/demand_per_step_prediction gives predicted surplus/demand; per step, per agent"""
            surplus_prediction = np.zeros((self.prediction_range, len(self.agents)))
            demand_per_step_prediction = np.zeros((self.prediction_range, len(self.agents)))

            """ index prediction data, now using the current data set... AI should be plugged in here eventually"""
            for i in range(self.prediction_range):
                for agent in self.agents[:]:
                    """ now using actual data (big_data_file) but should be substituted with prediction data"""
                    energy_per_step_per_agent_prediction = self.big_data_file[self.steps + i][agent.id][0] - self.big_data_file[self.steps + i][agent.id][1]         # [0] = load, corresponds with demand  - [1] = production
                    """ results in either demand_per_step_prediction or surplus_prediction for each agent"""
                    if energy_per_step_per_agent_prediction >= 0:
                        """ series of coming demands"""
                        demand_per_step_prediction[i][agent.id] = abs(energy_per_step_per_agent_prediction)
                    if energy_per_step_per_agent_prediction < 0:
                        """ series of coming surpluses"""
                        surplus_prediction[i][agent.id] = abs(energy_per_step_per_agent_prediction)


            """ for now this is the prediction """
            self.E_total_surplus_prediction_per_step = np.zeros(self.prediction_range) # must adapt in size for every step (shrinking as time progresses))
            self.E_total_demand_prediction = np.zeros(self.prediction_range)

            for i in range(self.prediction_range):
                self.E_total_surplus_prediction_per_step[i] = calc_E_total_prediction(surplus_prediction[i][:], horizon, N, self.steps, self.prediction_range)    # linear

            # plt.plot(self.E_total_surplus_prediction_per_step)
            # plt.show()
            """ A prediction"""
            for i in range(len(agent.comm_reach)):
                agent.R_total_list_masked[i] = self.R_total_list[agent.comm_reach[i]]
                agent.R_total = sum(agent.R_total_list_masked)
            for agent in self.agents[:]:
                agent.R_prediction, agent.alpha, agent.beta = calc_R_prediction_masked(agent.R_total, self.big_data_file, horizon,
                                                                   self.agents, agent.comm_reach, self.steps)

            """ conversion to usable E_supply_prediction (using w_prediction_avg does nothing yet)
                Make use of agents knowledge that is shared among each others: 
                total predicted energy = sum(individual predictions)"""


            """ E_surplus prediction over horizon in total = a sum of agents predictions"""
            self.w_prediction_avg = 0
            self.E_surplus_prediction_over_horizon = 0
            for agent in self.agents[:]:
                self.E_surplus_prediction_over_horizon_list[agent.id] = agent.E_surplus_prediction_agent
                self.w_prediction_avg += agent.w_prediction/N

            self.E_supply_prediction = sum(self.E_surplus_prediction_over_horizon_list) * self.w_prediction_avg
            """ analysis of prediction data """
            means_surplus = []
            means_load = []

            for i in range(self.prediction_range):
                means_surplus.append(np.mean(surplus_prediction[i][:]))
                means_load.append(np.mean(demand_per_step_prediction[i][:]))


            supply_old = self.E_total_supply

            """sellers-level game optimization"""
            while True: # sellers
                iteration_sellers += 1
                if len(self.sellers_pool) != 0 and sum(self.E_total_demand_list) != 0:
                    self.num_seller_iteration += 1

                for agent in self.agents[:]:
                    """ determine """
                    agent.E_supply_others = sum(self.E_total_supply_list) - self.E_total_supply_list[agent.id]
                    agent.bidding_prices_others = sum(self.c_bidding_prices) - agent.c_i_bidding_price
                    agent.E_supply_others_prediction = self.E_supply_prediction - agent.E_supply_prediction_agent  #+ self.battery_soc_total - agent.battery_capacity_n

                for agent in self.agents[:]:
                    if agent.classification == 'seller':


                        for i in range(len(agent.comm_reach)):
                            agent.R_total_list_masked[i] = self.R_total_list[agent.comm_reach[i]] # done
                        agent.R_total = sum(agent.R_total_list_masked)
                        agent.R_total = self.R_total_list[agent.comm_reach].sum()
                        agent.E_supply_others_direct = self.E_total_supply_list[agent.comm_reach].sum() - agent.E_j_supply
                        agent.E_supply_others_prediction = (self.E_surplus_prediction_over_horizon_list[agent.comm_reach].sum() - agent.E_supply_prediction_agent) * self.w_prediction_avg
                        """Sellers optimization game, plugging in bidding price to decide on sharing factor.
                        Is bidding price lower than that the smart-meter expects to sell on a later time period?
                        smart-meter needs a prediction on the coming day. Either use the load data or make a predicted model on 
                        all aggregated load data"""
                        prev_wj = agent.w_j_storage_factor
                        agent.bidding_prices_others = sum(self.c_bidding_prices)

                        """Optimization"""
                        sol_seller, \
                        sol_seller.x[0], \
                        utility_seller_function = sellers_game_optimization(agent.id,
                                                                                agent.E_j_surplus,
                                                                                agent.R_total,
                                                                                agent.E_supply_others,
                                                                                agent.R_prediction,
                                                                                agent.E_supply_others_prediction,
                                                                                agent.w_j_storage_factor,
                                                                                agent.E_supply_prediction_agent,
                                                                                agent.lower_bound_on_w_j,
                                                                                lambda_set)

                        agent.w_j_storage_factor = sol_seller.x[0]
                        prediction_utility, direct_utility, agent.utility_j = calc_utility_function_j(agent.id,
                                                                                                    agent.E_j_surplus,
                                                                                                    agent.R_total,
                                                                                                    agent.E_supply_others,
                                                                                                    agent.R_prediction,
                                                                                                    agent.E_supply_others_prediction,
                                                                                                    agent.w_j_storage_factor,
                                                                                                    agent.E_supply_prediction_agent, lambda_set)
                        if abs(agent.utility_j - sol_seller.fun) > 10:
                            sys.exit("utility_j calculation does not match with optimization code")
                            pass

                        agent.utility_seller = [agent.utility_j, prediction_utility, direct_utility]
                        self.utilities_sellers[agent.id] = agent.utility_seller

                        """ Update on values"""
                        agent.E_j_supply = agent.E_j_surplus * agent.w_j_storage_factor
                        self.E_total_supply_list[agent.id] = agent.E_j_supply
                        if self.E_total_supply_list[agent.id] <= 0 or np.isnan(self.E_total_supply_list[agent.id]):
                            self.E_total_supply_list[agent.id] = 0

                        self.E_total_supply = sum(self.E_total_supply_list)
                        agent.E_supply_others = sum(self.E_total_supply_list) - self.E_total_supply_list[agent.id]
                        self.w_storage_factors[agent.id] = agent.w_j_storage_factor

                        """Analyses"""
                        difference_rev = self.R_total - self.R_prediction
                        difference_load = self.E_total_supply - self.E_surplus_prediction_over_horizon

                        """Tolerance"""
                        new_wj = agent.w_j_storage_factor
                        tolerance_sellers[agent.id] = abs(new_wj - prev_wj)
                    else:
                        agent.w_j_storage_factor = 0


                self.E_total_supply = sum(self.E_total_supply_list)
                self.E_demand = sum(self.E_total_demand_list)
                # self.R_total_end = self.E_total_supply * self.c_nominal
                for agent in self.agents[:]:
                    agent.E_supply_others = sum(self.E_total_supply_list) - self.E_total_supply_list[agent.id]

                self.w_nominal = sum(self.E_total_supply_list) / self.E_total_surplus

                epsilon_sellers_game = max(abs(tolerance_sellers))
                if epsilon_sellers_game < e_sellers:
                    tolerance_sellers[:] = 0
                    supply_new = self.E_total_supply
                    """ Update on batteries """
                    for agent in self.agents[:]:
                        self.actual_batteries[agent.id] = agent.soc_actual
                        # print(self.actual_batteries)
                    break
                else:
                    pass

            tolerance_supply = abs(supply_new - supply_old)
            epsilon_c_nominal = abs(tolerance_c_nominal)
            if (epsilon_c_nominal < e_cn and tolerance_supply < e_supply) or iteration_global > 30:
                break


        self.deficit_total = 0

        total_payment = 0
        """settle all deals"""
        for agent in self.agents[:]:
            agent.payment = 0
            if self.steps == 75:
                print('hello')
            if agent.classification == 'buyer':
                """ buyers costs """
                agent.E_i_allocation = allocation_i(self.E_total_supply, agent.c_i_bidding_price, agent.bidding_prices_others)
                self.supply_deals[agent.id] = agent.E_i_allocation
                agent.soc_influx = agent.E_i_allocation - agent.E_i_demand
                if agent.soc_actual + agent.soc_influx < 0:
                    """ battery is depleting """
                    agent.deficit = agent.soc_actual + agent.soc_influx
                    agent.soc_influx = agent.E_i_demand + agent.deficit
                    self.deficit_total += agent.deficit
                if agent.soc_actual + agent.soc_influx > agent.battery_capacity_n:
                    """ battery is overflowing """
                    agent.soc_influx = agent.batt_available - agent.soc_actual
                    agent.batt_overflow = agent.soc_actual + agent.soc_influx - agent.batt_available
                agent.soc_actual += agent.soc_influx
                agent.payment = agent.E_i_allocation * agent.c_i_bidding_price
                total_payment += agent.payment
            self.actual_batteries[agent.id] = agent.soc_actual

        for agent in self.agents[:]:
            agent.revenue = 0
            if agent.classification == 'seller':
                """ sellers earnings """
                if self.steps == 75:
                    print('hello')
                agent.soc_influx = agent.E_j_surplus * (1 - agent.w_j_storage_factor) + agent.E_j_returned_supply
                agent.revenue = agent.E_j_supply/self.E_total_supply * total_payment # E_j_actual_supplied
                agent.soc_actual += agent.soc_influx
            self.actual_batteries[agent.id] = agent.soc_actual
        if np.any(self.actual_batteries) < 0 or np.any(self.actual_batteries) > agent.battery_capacity_n:
            exit("negative battery soc, physics are broken")

        self.battery_soc_total = sum(self.actual_batteries)

        self.payed_list = np.zeros(N)
        self.received_list = np.zeros(N)

        for agent in self.agents[:]:
            self.payed_list[agent.id] = agent.payment
            self.received_list[agent.id] = agent.revenue

        """ Pull data out of agents """
        total_soc_pref = 0
        for agent in self.agents[:]:
            total_soc_pref += agent.soc_preferred
            self.soc_preferred[agent.id] = agent.soc_preferred
        avg_soc_preferred = total_soc_pref/N

        """ Blockchain """
        if blockchain == 'on':
            for agent in self.agents[:]:
                agent.settlement_of_payments(self.w3, self.contract_instance)

        self.profit_list = np.zeros(N)
        """ Costs on this step"""
        for agent in self.agents[:]:
            agent.profit = agent.revenue - agent.payment
            self.profit_list[agent.id] = agent.profit

        """ Update time """
        self.steps += 1
        self.time += 1
        return self.E_total_surplus, self.E_total_supply, self.E_demand, \
               self.buyers_pool, self.sellers_pool, self.w_storage_factors, \
               self.c_nominal, self.w_nominal, \
               self.R_prediction, self.E_supply_prediction, self.E_total_supply, self.R_total, \
               self.actual_batteries, self.E_total_supply, self.E_demand, \
               self.utilities_buyers, self.utilities_sellers, \
               self.soc_preferred, avg_soc_preferred, \
               self.E_total_demand_list, self.c_bidding_prices, self.E_surplus_list, \
               self.num_global_iteration,self.num_buyer_iteration, self.num_seller_iteration, TODOprofit_list

