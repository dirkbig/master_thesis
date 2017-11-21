from source.BatteryModel import *
from source.PvModel import *
import pandas as pd
from source.function_file import *
from source.initialization import *
from mesa import Agent, Model


"""All starting parameters are initialised"""
duration = 1  # 1440                # Duration of the sim (one full day or 1440 time_steps of 1 minute) !!10 if test!!
N = 5                               # N agents only
step_list = np.zeros([duration])

c_S = 4                                              # c_S is selling price of the microgrid
c_B = 2                                              # c_B is buying price of the microgrid
c_macro = (c_B, c_S)                                 # Domain of available prices for bidding player i
possible_c_i = range(c_macro[0], c_macro[1])         # domain of solution for bidding price c_i



""" Household class """

class HouseholdAgent(Agent):
    """All microgrid household(agents) should be generated here; initialisation of prosumer tools """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        """agent characteristics"""
        self.id = unique_id
        self.battery_capacity = 10                                          # everyone has 10 kwh battery
        self.pv_generation = random.uniform(0, 1)                           # random.choice(range(15)) * pvgeneration
        self.consumption = random.uniform(0, 1)                             # random.choice(range(15)) * consumption
        self.classification = []

        """control variables"""
        self.gamma = initialize("gamma")                                    # sellers
        self.E_j_surplus = initialize("E_j_surplus")                        # sellers
        self.E_i_demand = initialize("E_i_demand")                          # buyers
        self.used_energy = initialize("used_energy")                        # both
        self.E_i_allocation = initialize("E_i_allocation")                  # buyers
        self.stored_energy = initialize("stored_energy")                    # both
        self.available_storage = initialize("available_storage")            # both
        self.payment_to_seller = initialize("payment_to_seller")            # buyer
        self.w_j_storage_factor = 0
        self.E_j_supply = 0
        self.c_i_bidding_price = 0


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
        classification = define_pool(self.consumption, self.pv_generation)
        self.classification = pool[0]



















