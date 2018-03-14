from blockchain.smartcontract import *
from source.function_file import *
import sys
import numpy as np
from mesa import Agent, Model


#####################################
### SYNCHRONOUS MODEL NON-TRADING ###
#####################################

"""All starting parameters are initialised"""
starting_point = 0
stopping_point = 7200 - starting_point - 200
step_day = 1440
timestep = 5
days = 5
blockchain = 'off'

step_time = 10
total_steps = step_day*days
sim_steps = int(total_steps/step_time)

average_consumption_household = 4000/365 #kWh/day
average_production_solarpanel = 15
comm_radius = 10

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
    def __init__(self, unique_id, model, w3, addr):
        super().__init__(unique_id, model)

        """agent characteristics"""
        self.id = unique_id
        self.battery_capacity_n = 150                           # every household has an identical battery, for now
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
        self.max_horizon =  700
        self.horizon_agent = min(self.max_horizon, sim_steps - self.current_step)  # including current step
        self.predicted_E_surplus_list = np.zeros(self.horizon_agent)
        self.w_j_prediction = 0.5
        """Battery related"""
        self.soc_actual = 0
        self.soc_preferred = self.soc_actual * 0.7

        self.soc_gap = 0
        self.soc_influx = 0
        self.batt_available = self.battery_capacity_n - self.soc_actual

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

        self.soc_actual = random.uniform(0.5 * self.soc_actual, self.battery_capacity_n)
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
        self.pv_generation = big_data_file_per_step[self.id, 1]           # import generation file

        """ agents personal prediction, can be different per agent (difference in prediction quality?)"""
        self.horizon_agent = min(self.max_horizon, sim_steps - self.current_step - 1)  # including current step
        self.predicted_E_surplus_list = np.zeros(self.horizon_agent)

        for i in range(self.horizon_agent):
            self.predicted_E_surplus_list[i] = big_data_file[steps + i][self.id][0] \
                                               - big_data_file[steps + i][self.id][1]  # load - production
            if self.predicted_E_surplus_list[i] < 0:
                self.predicted_E_surplus_list[i] = 0

        self.w_prediction = calc_w_prediction() # has to go to agent
        self.E_prediction_agent = calc_E_surplus_prediction(self.predicted_E_surplus_list,
                                                            self.horizon_agent, N, prediction_range, steps) # from surplus

        self.E_prediction_agent = self.E_prediction_agent * self.w_prediction # to actual supply

        """Determine state of charge of agent's battery"""
        self.current_step = steps
        battery_horizon = self.horizon_agent  # including current step

        self.soc_preferred = get_preferred_soc(self.soc_preferred, self.battery_capacity_n,
                                               self.predicted_E_surplus_list, self.soc_actual, battery_horizon)
        self.soc_gap = self.soc_preferred - self.soc_actual
        self.soc_surplus = 0
        if self.soc_gap < 0:
            self.soc_surplus = abs(self.soc_gap)
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
                self.soc_actual -= self.consumption
                self.action = 'self-supplying from battery'
            else:
                self.c_i_bidding_price = random.uniform(min(c_macro), max(c_macro))
                """values for sellers are set to zero"""
                self.action = 'bidding on the minutes-ahead market'
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
                self.action = 'selling to the grid'
            elif self.batt_available < self.E_j_surplus:
                """the part that HAS to be sold otherwise the battery is filled up to its max"""
                self.lower_bound_on_w_j = (self.E_j_surplus - self.batt_available)/self.E_j_surplus
                self.w_j_storage_factor = random.uniform(self.lower_bound_on_w_j, 0.8)
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

class MicroGrid_sync_not_trading(Model):


    """create environment in which agents can operate"""
    def __init__(self, big_data_file, starting_point, N, lambda_set):
        print("Synchronous Model")
        """Initialization and automatic creation of agents"""

        self.steps = starting_point
        self.time = starting_point

        self.num_households = N
        self.steps = 0
        self.time = 0
        self.big_data_file = big_data_file
        self.agents = []
        self.E_consumption_list  = np.zeros(N)
        self.E_production_list  = np.zeros(N)
        self.E_total_supply = 0
        self.E_total_surplus = 0
        self.c_bidding_prices = np.zeros(N)
        self.c_nominal = 0
        self.w_storage_factors = np.zeros(N)
        self.w_nominal = 0
        self.R_total = 0
        self.E_demand = 0
        self.E_allocation_list = np.zeros(N)
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
        self.E_prediction = 0
        self.E_supply_prediction = 0
        self.E_surplus_prediction_over_horizon = 0
        self.w_prediction_avg = 0.5
        self.c_nominal_prediction = 0
        self.E_total_demand_prediction = 0
        self.E_surplus_prediction_per_agent = 0
        self.soc_preferred_list = np.zeros(N)

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

        """ settlement """
        self.supply_deals = np.zeros(N)
        self.E_surplus_list = np.zeros(N)
        self.profit_list = np.zeros(N)

        self.num_global_iteration = 0
        self.num_buyer_iteration = 0
        self.num_seller_iteration = 0

        self.payed_list = np.zeros(N)
        self.received_list = np.zeros(N)
        self.deficit_total = 0
        self.deficit_total_progress = 0
        self.E_actual_supplied_list = np.zeros(N)
        self.E_allocation_list = np.zeros(N)

        self.revenue_list = np.zeros(N)
        self.payment_list = np.zeros(N)

    def step(self, N, lambda_set):
        """Environment proceeds a step after all agents took a step"""
        print("Step =", self.steps)

        for agent in self.agents[:]:
            self.actual_batteries[agent.id] = agent.soc_actual

        self.utilities_sellers = np.zeros((N, 3))
        self.utilities_buyers = np.zeros((N, 4))

        # random.shuffle(self.agents)
        for agent in self.agents[:]:
            self.E_consumption_list[agent.id] = agent.pv_generation
            self.E_production_list[agent.id] = agent.consumption

        self.sellers_pool = []
        self.buyers_pool = []
        self.passive_pool = []

        self.E_surplus_list = np.zeros(N)
        self.E_total_supply_list = np.zeros(N)
        self.E_demand_list = np.zeros(N)

        self.E_total_surplus = 0
        self.E_total_supply = 0

        self.num_global_iteration = 0
        self.num_buyer_iteration = 0
        self.num_seller_iteration = 0

        """DYNAMICS"""
        """ determine length/distance of horizon over which prediction data is effectual through a weight system (log, linear.. etc) """
        horizon = int(min(sim_steps/days, sim_steps - self.steps))  # including current step

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
                self.E_total_supply_list[agent.id] = agent.E_j_supply
                self.c_bidding_prices[agent.id] = 0
                self.E_demand_list[agent.id] = 0
            else:
                self.passive_pool.append(agent.id)

        self.E_total_surplus = sum(self.E_surplus_list)
        self.E_total_demand = sum(self.E_demand_list)

        if np.any(np.isnan(self.E_total_supply_list)):
            print("some supply is NaN!?")
            for agent in self.agents[:]:
                if np.isnan(self.E_total_supply_list[agent.id]):
                    self.E_total_supply_list[agent.id] = 0

        self.E_total_supply = sum(self.E_total_supply_list)
        for agent in self.agents[:]:
            agent.E_supply_others = sum(self.E_total_supply_list) - self.E_total_supply_list[agent.id]
            # self.E_allocation_per_agent[agent.id] = agent.E_i_allocation

        """Optimization after division of players into pools"""
        self.c_nominal = 0
        self.R_total = 0
        self.R_prediction = 0
        self.w_nominal = 0


        for agent in self.agents[:]:
            agent.E_i_allocation = 0
            agent.E_j_supply = 0
            agent.E_j_actual_supplied = 0
            agent.c_i_bidding_price = 0
            agent.w_j_storage_factor = 0
        self.deficit_total = 0
        self.supply_deals = np.zeros(N)
        """settle all deals"""

        for agent in self.agents[:]:
            agent.deficit = 0
            agent.influx = agent.pv_generation - agent.consumption
            if agent.soc_actual + agent.influx <= 0:
                """ depletion """
                agent.soc_actual = 0
                agent.deficit = abs(agent.soc_actual + agent.influx)
                agent.influx = - agent.soc_actual
            if agent.soc_actual + agent.influx >= agent.battery_capacity_n:
                """ overflow """
                agent.soc_actual = agent.battery_capacity_n
                agent.influx = agent.battery_capacity_n - agent.soc_actual
            agent.soc_actual += agent.influx
            self.deficit_total += agent.deficit
        if np.any(self.actual_batteries) < -0.01 or np.any(self.actual_batteries) > agent.battery_capacity_n + 0.01:
            exit("negative battery soc, physics are broken")


        self.deficit_total_progress += self.deficit_total
        self.battery_soc_total = sum(self.actual_batteries)

        """ Pull data out of agents """
        total_soc_pref = 0
        for agent in self.agents[:]:
            total_soc_pref += agent.soc_preferred
            self.soc_preferred_list[agent.id] = agent.soc_preferred
        avg_soc_preferred = total_soc_pref/N

        """ Blockchain """

        self.profit_list = np.zeros(N)
        """ Costs on this step"""
        self.profit_list = np.zeros(N)
        self.revenue_list = np.zeros(N)
        self.payed_list = np.zeros(N)

        """ Update time """
        self.steps += 1
        self.time += 1

        return self.E_total_surplus, self.E_demand, \
               self.buyers_pool, self.sellers_pool, self.w_storage_factors, \
               self.c_nominal, self.w_nominal, \
               self.R_prediction, self.E_supply_prediction, self.R_total, \
               self.actual_batteries, self.E_total_supply, \
               self.utilities_buyers, self.utilities_sellers, \
               self.soc_preferred_list, avg_soc_preferred, \
               self.E_consumption_list, self.E_production_list, \
               self.E_demand_list, self.c_bidding_prices, self.E_surplus_list, self.E_total_supply_list,\
               self.num_global_iteration, self.num_buyer_iteration, self.num_seller_iteration, \
               self.profit_list, self.revenue_list, self.payment_list,\
               self.deficit_total, self.deficit_total_progress, self.E_actual_supplied_list, self.E_allocation_list



    def __repr__(self):
        return "no trading Microgrid"

