from blockchain.smartcontract import *
from source.function_file import *
import sys
import numpy as np
from mesa import Agent, Model
from pyswarm import pso

###########################################
### PSO HIERARCHICAL OPTIMIZATION MODEL ###
###########################################
print("PSO HIERARCHICAL OPTIMIZATION MODEL")

"""All starting parameters are initialised"""
starting_point = 0
stopping_point = 7200 - starting_point - 1000
step_day = 1440
timestep = 5
days = 5
blockchain = 'off'

step_time = 10
total_steps = step_day*days
sim_steps = int(total_steps/step_time)

comm_radius = 10

step_list = np.zeros([sim_steps])

c_S = 10                                             # c_S is selling price of the microgrid
c_B = 1                                              # c_B is buying price of the microgrid
c_macro = (c_B, c_S)                                 # Domain of available prices for bidding player i
possible_c_i = range(c_macro[0], c_macro[1])         # domain of solution for bidding price c_i

e_buyers = 0.001
e_sellers = 0.0001
e_global = 0.0001
e_avg_buyers = 1
e_avg_sellers = 0.001

class HouseholdAgent(Agent):


    """All microgrid household(agents) should be generated here; initialisation of prosumer tools """
    def __init__(self, unique_id, model, w3, addr):
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

        self.E_prediction_agent = 0
        self.w_prediction = 0
        self.solar_capacity = 0

        self.soc_actual = random.uniform(0.5 * self.soc_actual, 0.8 * self.soc_actual)
        self.solar_capacity = 1

        self.revenue = 0
        self.payment = 0

        """Blockchain"""
        if blockchain == 'on':
            self.address_agent = w3.eth.accounts[addr]

        self.promise_on_bc = 0
        self.balance_on_bc = 0

        self.profit = 0
        self.E_j_actual_supplied = 0
        self.E_j_returned_supply = 0

    def step(self, big_data_file_per_step, big_data_file, E_total_surplus_prediction_per_step, horizon, prediction_range, steps, w3, contract_instance, N):           # big_data_file = np.zeros((N, step_time, 3))
        """Agent optimization step, what ever specific agents do on during step"""

        self.E_j_actual_supplied = 0
        self.E_j_returned_supply = 0

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
        self.E_prediction_agent = calc_E_surplus_prediction(self.predicted_E_surplus_list,
                                                            self.horizon_agent, N, prediction_range, steps) # from surplus

        self.E_prediction_agent = self.E_prediction_agent * self.w_prediction # to actual supply

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


    def settlement_of_payments(self, w3, contract_instance, N):
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
            self.balance_on_bc = setter_mint(w3, contract_instance, self.address_agent, int(self.revenue))
            self.payment = 0


        # print('balance of agent %d = %d' % (self.id, self.balance_on_bc))
        return


    def __repr__(self):
        return "ID: %d, batterycapacity:%d, pvgeneration:%d, consumption:%d" % (self.id, self.battery_capacity, self.pv_generation, self.consumption)


"""Microgrid model environment"""

class MicroGrid_PSO(Model):


    """create environment in which agents can operate"""
    def __init__(self, big_data_file, starting_point, N, lambda_set):
        print("MicroGrid_PSO")
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
        self.E_supply_per_agent = np.zeros(N)
        self.E_demand = 0
        self.E_allocation_per_agent = np.zeros(N)
        self.E_total_supply_list = np.zeros(N)
        self.E_demand_list = np.zeros(N)
        self.E_allocation_total = 0
        self.E_total_demand = 0
        self.E_demand_list = np.zeros(N)
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
        self.R_prediction_list = np.zeros(N)
        self.E_prediction = 0
        self.E_supply_prediction = 0
        self.E_surplus_prediction_over_horizon = 0
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
            agent = HouseholdAgent(i, self, self.w3, addr)
            self.agents.append(agent)

        if blockchain == 'on':
            for agent in self.agents[:]:
                setter_initialise_tokens2(self.w3, self.contract_instance, deployment_tx_hash, agent.address_agent)

        self.E_prediction_list = np.zeros(N)

        """ settlement """
        self.supply_deals = np.zeros(N)
        self.E_surplus_list = np.zeros(N)
        self.E_total_allocation = np.zeros(N)
        self.profit_list = np.zeros(N)

        self.num_global_iteration = 0
        self.num_buyer_iteration = 0
        self.num_seller_iteration = 0

        self.lb_buyer = 0
        self.ub_buyer = 1000

        self.lb_seller = 0
        self.ub_seller = 1

    def step(self, N, lambda_set):
        """Environment proceeds a step after all agents took a step"""
        print("Step =", self.steps)

        for agent in self.agents[:]:
            self.actual_batteries[agent.id] = agent.soc_actual

        self.utilities_sellers = np.zeros((N, 3))
        self.utilities_buyers = np.zeros((N, 4))

        # random.shuffle(self.agents)
        self.sellers_pool = []
        self.buyers_pool = []
        self.passive_pool = []

        self.E_surplus_list = np.zeros(N)
        self.E_total_supply_list = np.zeros(N)
        self.E_demand_list = np.zeros(N)

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
            agent.step(self.big_data_file[self.steps], self.big_data_file, self.E_total_surplus_prediction_per_step, horizon, self.prediction_range, self.steps, self.w3, self.contract_instance, N)
            if agent.classification == 'buyer':
                """Level 1 init game among buyers"""
                self.buyers_pool.append(agent.id)
                self.c_bidding_prices[agent.id] = agent.c_i_bidding_price
                self.E_demand_list[agent.id] = agent.E_i_demand
                agent.w_j_storage_factor = 0
                agent.E_j_surplus = 0
                agent.E_j_supply = 0
            elif agent.classification == 'seller':
                """Level 2 init of game of sellers"""
                self.sellers_pool.append(agent.id)
                self.w_storage_factors[agent.id] = agent.w_j_storage_factor
                self.E_surplus_list[agent.id] = agent.E_j_surplus
                agent.E_j_supply = agent.E_j_surplus * agent.w_j_storage_factor
                self.E_total_supply_list[agent.id] = agent.E_j_supply     # does change by optimization
                self.c_bidding_prices[agent.id] = 0
                self.E_demand_list[agent.id] = 0
            else:
                self.passive_pool.append(agent.id)

        self.E_total_surplus = sum(self.E_surplus_list)                               # does not change by optimization
        self.E_total_demand = sum(self.E_demand_list)

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


        """global tolerance"""
        w_nominal_list = [0]
        c_nominal_list = [0]
        w_nominal_avg_list = [0]
        c_nominal_avg_list = [0]

        while True: # global
            if len(self.buyers_pool) != 0 and len(self.sellers_pool) != 0:
                self.num_global_iteration += 1
            prev_c_nominal = self.c_nominal
            # print("total energy available to buyers is ", self.E_total_supply)
            if self.E_total_supply == 0:
                for agent in self.agents[:]:
                    agent.E_i_allocation = 0
                    self.supply_deals[agent.id] = agent.E_i_allocation
                # print("no energy available in community this time")
                break


            """Buyers level optimization"""
            iteration_buyers += 1
            if len(self.buyers_pool) != 0 and sum(self.E_surplus_list) != 0:
                self.num_buyer_iteration += 1
            buyers_mask = np.ones(N)
            for agent in self.agents[:]:
                if agent.classification == 'buyer':
                    buyers_mask[agent.id] = 0
                else:
                    pass
            """agent.c_i_bidding_price should incorporate its E_i_demand. If allocation is lower than the actual E_i_demand, payment to macro-grid 
            (which will be more expensive than buying from Alice) needs to be minimised. 
            Utility of buyer should be: (total energy demand - the part allocated from alice(c_i) ) * c_macro
            then allocation will be increased by offering more money"""
            """ PSO BUYERS"""
            lambda11_list = np.ones(N)*lambda_set[0]
            lambda12_list = np.ones(N)*lambda_set[1]

            i = 0
            N_buyers = len(self.buyers_pool)
            E_demand_list_buyer = np.zeros(N_buyers)
            lambda11_list_buyer = np.zeros(N_buyers)
            lambda12_list_buyer = np.zeros(N_buyers)
            for agent in self.agents[:]:
                if agent.classification == 'buyer':
                    E_demand_list_buyer[i] = self.E_demand_list[agent.id]
                    lambda11_list_buyer[i] = lambda11_list[agent.id]
                    lambda12_list_buyer[i] = lambda12_list[agent.id]
                    i += 1

            lb_buyer = np.zeros(N_buyers)
            ub_buyer = np.zeros(N_buyers)
            for i in range(N_buyers):
                lb_buyer[i] = self.lb_buyer
                ub_buyer[i] = self.ub_buyer

            args_buyer = [E_demand_list_buyer, self.E_total_supply, lambda11_list_buyer, lambda12_list_buyer, N_buyers]
            c_opt, f_opt = pso(buyer_objective_function_PSO, lb_buyer, ub_buyer, ieqcons=[], args=args_buyer, swarmsize=1000,
                               omega=0.9, phip=0.6,
                               phig=0.6, maxiter=1000, minstep=1e-1, minfunc=1e-1)  #
            print(c_opt)

            self.c_bidding_prices = np.zeros(N)
            for i in range(len(self.buyers_pool)):
                self.c_bidding_prices[self.buyers_pool[i]] = c_opt[i]

            """ Update on buyer values"""
            for agent in self.agents[:]:
                agent.c_i_bidding_price = self.c_bidding_prices[agent.id]
                agent.bidding_prices_others = bidding_prices_others(self.c_bidding_prices, agent.c_i_bidding_price)
                agent.E_i_allocation = allocation_i(self.E_total_supply, agent.c_i_bidding_price, agent.bidding_prices_others)
                payment_to_seller[agent.id] = agent.c_i_bidding_price * agent.E_i_allocation
                self.E_allocation_per_agent[agent.id] = agent.E_i_allocation


            self.c_nominal = sum(self.E_allocation_per_agent * self.c_bidding_prices) / sum(self.E_allocation_per_agent)
            if sum(self.E_allocation_per_agent) == 0:
                self.c_nominal = 0
            self.R_total = sum(payment_to_seller)
            self.E_allocation_total = sum(self.E_allocation_per_agent)

            """Determine global tolerances"""
            self.c_nominal = np.dot(self.E_allocation_per_agent , self.c_bidding_prices) / sum(self.E_allocation_per_agent)
            c_average = sum(self.c_bidding_prices)/N_buyers

            if np.isnan(self.c_nominal):
                exit("c_i is NaN")
                pass
            new_c_nominal = self.c_nominal
            tolerance_c_nominal = abs(new_c_nominal - prev_c_nominal)
            c_nominal_list = np.append(c_nominal_list, self.c_nominal)

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

            """ A prediction"""
            for agent in self.agents[:]:
                self.R_prediction, alpha, beta = calc_R_prediction(self.R_total, self.big_data_file, horizon, self.agents, self.steps)
                self.R_prediction_list[agent.id] = self.R_prediction
            """ E_surplus prediction over horizon in total = a sum of agents predictions"""
            self.E_surplus_prediction_over_horizon = 0
            for agent in self.agents[:]:
                self.E_surplus_prediction_over_horizon += agent.E_prediction_agent
                self.E_prediction_list[agent.id] = agent.E_prediction_agent

            """ conversion to usable E_supply_prediction (using w_prediction_avg does nothing yet)
                Make use of agents knowledge that is shared among each others: 
                total predicted energy = sum(individual predictions)"""
            self.w_prediction_avg = 0
            for agent in self.agents[:]:
                self.w_prediction_avg += agent.w_prediction/N


            self.E_supply_prediction = self.E_surplus_prediction_over_horizon * self.w_prediction_avg

            """ analysis of prediction data """
            means_surplus = []
            means_load = []

            for i in range(self.prediction_range):
                means_surplus.append(np.mean(surplus_prediction[i][:]))
                means_load.append(np.mean(demand_per_step_prediction[i][:]))

            """sellers-level game optimization"""
            iteration_sellers += 1
            if len(self.sellers_pool) != 0 and sum(self.E_demand_list) != 0:
                self.num_seller_iteration += 1

            for agent in self.agents[:]:
                agent.E_supply_others = sum(self.E_total_supply_list) - self.E_total_supply_list[agent.id]
                agent.bidding_prices_others = sum(self.c_bidding_prices) - agent.c_i_bidding_price
                agent.E_supply_others_prediction = self.E_supply_prediction - agent.E_prediction_agent  #+ self.battery_soc_total - agent.battery_capacity_n

            """ PSO SELLERS"""
            """ sellers level game """
            R_direct = np.dot(self.E_demand_list, self.c_bidding_prices)

            lambda21_list = np.ones(N)*lambda_set[2]
            lambda22_list = np.ones(N)*lambda_set[3]

            N_seller = len(self.sellers_pool)
            E_surplus_list_seller = np.zeros(N_seller)
            R_future_list_seller = np.zeros(N_seller)
            E_predicted_list_seller = np.zeros(N_seller)
            lambda21_list_seller = np.zeros(N_seller)
            lambda22_list_seller = np.zeros(N_seller)

            j = 0
            for agent in self.agents[:]:
                if agent.classification == 'seller':
                    E_surplus_list_seller[j] = self.E_surplus_list[agent.id]
                    R_future_list_seller[j] = self.R_prediction_list[agent.id]
                    E_predicted_list_seller[j] = self.E_prediction_list[agent.id]
                    lambda21_list_seller[j] = lambda21_list[agent.id]
                    lambda22_list_seller[j] = lambda22_list[agent.id]
                    j += 1

            lb_seller = np.zeros(N_seller)
            ub_seller = np.zeros(N_seller)

            for j in range(N_seller):
                lb_seller[j] = self.lb_seller
                ub_seller[j] = self.ub_seller

            args_seller = [R_direct, E_surplus_list_seller, R_future_list_seller, E_predicted_list_seller, lambda21_list_seller, lambda22_list_seller, N_seller]

            w_opt, f_opt = pso(seller_objective_function_PSO, lb_seller, ub_seller, ieqcons=[], args=args_seller,
                               swarmsize=1000, omega=0.9, phip=0.6,
                               phig=0.6, maxiter=1000, minstep=1e-1, minfunc=1e-1)  #
            print("w_opt")
            print(w_opt)

            for i in range(len(self.sellers_pool)):
                self.w_storage_factors[self.sellers_pool[i]] = w_opt[i]

            """ Update on values"""
            for agent in self.agents[:]:
                agent.w_j_storage_factor = self.w_storage_factors[agent.id]
                agent.E_j_supply = agent.E_j_surplus * agent.w_j_storage_factor
                self.E_total_supply_list[agent.id] = agent.E_j_supply
                if self.E_total_supply_list[agent.id] <= 0 or np.isnan(self.E_total_supply_list[agent.id]):
                    self.E_total_supply_list[agent.id] = 0
                agent.E_supply_others = sum(self.E_total_supply_list) - self.E_total_supply_list[agent.id]
                self.w_storage_factors[agent.id] = agent.w_j_storage_factor

            self.E_total_supply = sum(self.E_total_supply_list)
            self.E_demand = sum(self.E_demand_list)
            self.w_nominal = sum(self.E_total_supply_list) / self.E_total_surplus
            w_nominal_list = np.append(w_nominal_list, self.w_nominal)
            print('self.w_nominal', self.w_nominal)


            self.E_total_supply = sum(self.E_total_supply_list)
            for agent in self.agents[:]:
                self.actual_batteries[agent.id] = agent.soc_actual


            tolerance_sellers = abs(w_nominal_list[-1] - w_nominal_list[-2])
            tolerance_buyers = abs(c_nominal_list[-1] - c_nominal_list[-2])
            iteration_global += 1

            w_nominal_avg = np.mean(w_nominal_list)
            c_nominal_avg = np.mean(c_nominal_list)
            w_nominal_avg_list = np.append(w_nominal_avg_list, w_nominal_avg)
            c_nominal_avg_list = np.append(c_nominal_avg_list, c_nominal_avg)

            tolerance_sellers_avg = abs(w_nominal_avg_list[-1] - w_nominal_avg_list[-2])
            tolerance_buyers_avg = abs(c_nominal_avg_list[-1] - c_nominal_avg_list[-2])
            """ END OF ROUND criteria """
            if (tolerance_buyers < e_buyers and tolerance_sellers < e_sellers) \
                    or (tolerance_buyers_avg < e_avg_buyers and tolerance_sellers_avg < e_avg_sellers) \
                    or self.num_global_iteration > 30:
                np.save('/Users/dirkvandenbiggelaar/Desktop/python_plots/np_arrays/w_nominal_list_' + str(self.steps), w_nominal_list)
                np.save('/Users/dirkvandenbiggelaar/Desktop/python_plots/np_arrays/c_nominal_list_' + str(self.steps), c_nominal_list)
                np.save('/Users/dirkvandenbiggelaar/Desktop/python_plots/np_arrays/w_nominal_avg_list' + str(self.steps), w_nominal_avg_list)
                np.save('/Users/dirkvandenbiggelaar/Desktop/python_plots/np_arrays/c_nominal_avg_list' + str(self.steps), c_nominal_avg_list)

                for agent in self.agents[:]:
                    self.E_allocation_per_agent[agent.id] = agent.E_i_allocation
                self.E_total_supply = sum(self.E_total_supply_list)
                self.E_total_allocation = sum(self.E_allocation_per_agent)
                ratio_oversupply = self.E_total_allocation/self.E_total_supply
                print('ratio_oversupply', ratio_oversupply)
                if ratio_oversupply > 1.0:
                    """ Sellers are offering more energy than demand on the market, profit is divided """
                    for agent in self.agents[:]:
                        agent.E_j_actual_supplied = agent.E_j_supply / ratio_oversupply
                        agent.E_j_returned_supply = agent.E_j_supply - agent.E_j_actual_supplied
                break
            else:
                pass




        """settle all deals"""
        for agent in self.agents[:]:
            if agent.classification == 'buyer':
                agent.E_i_allocation = allocation_i(self.E_total_supply, agent.c_i_bidding_price, agent.bidding_prices_others)
                self.supply_deals[agent.id] = agent.E_i_allocation
                agent.soc_influx = agent.E_i_allocation - agent.E_i_demand
                if agent.soc_actual + agent.soc_influx < 0.01:
                    # print("agent %d battery depleted" % agent.id)
                    agent.deficit = agent.soc_actual + agent.soc_influx
                    agent.soc_influx = agent.E_i_demand - agent.deficit
                if agent.soc_actual + agent.soc_influx > agent.batt_available:
                    agent.soc_influx = agent.batt_available - agent.soc_actual
                    agent.batt_overflow = agent.soc_actual + agent.soc_influx - agent.batt_available
                    # print("agent %d battery overflowing" % agent.id)
                agent.soc_actual += agent.soc_influx
                agent.payment = agent.E_i_allocation * agent.c_i_bidding_price
            elif agent.classification == 'seller':
                agent.soc_influx = agent.E_j_surplus * (1 - agent.w_j_storage_factor) + agent.E_j_returned_supply
                agent.revenue = agent.E_j_actual_supplied * agent.w_j_storage_factor * self.c_nominal
                agent.soc_actual += agent.soc_influx
            self.actual_batteries[agent.id] = agent.soc_actual
        if np.any(self.actual_batteries) < 0:
            exit("negative battery soc, physiscs are broken")
        self.battery_soc_total = sum(self.actual_batteries)

        total_payed = 0
        total_recieved = 0
        for agent in self.agents[:]:
            total_payed += agent.payment
            total_recieved += agent.revenue
        print('total_payed',total_payed)
        print('total_recieved' ,total_recieved)
        print('total_recieved - total_payed', total_recieved - total_payed)

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
               self.E_demand_list, self.c_bidding_prices, self.E_surplus_list, \
               self.num_global_iteration, self.num_buyer_iteration, self.num_seller_iteration, \
               self.profit_list

    def __repr__(self):
        return "PSO HIERARCHICAL OPTIMIZATION MODEL"

