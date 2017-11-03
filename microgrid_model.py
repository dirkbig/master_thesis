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
    def __init__(self, unique_id, batterycapacity_total, pvgeneration, consumption, model):
        super().__init__(unique_id, model)
        self.ID = unique_id
        self.BatteryCapacity = 0             # random.choice(range(15)) * batterycapacity_total
        self.PvGeneration = 0              # random.choice(range(15)) * pvgeneration
        self.Consumption = 0              # random.choice(range(15)) * consumption
        self.Classification = []

    def step(self, bla):
        """Agent optimization step, what ever specific agents do on during step"""
        self.PvGeneration = bla[unique_id][1]
        self.Consumption = 10
        self.BatteryCapacity = 0
        classification = define_pool(self.Consumption, self.PvGeneration)               # First define the pool
        if classification == "seller":
            print("I am a seller")
            self.Classification = ["buyer"]
        if classification == "buyer":
            print("I am a buyer")
            self.Classification = ["seller"]


    def seller_utility(self):
        """calculates sellers utility"""
        pass

    def buyer_utility(selfs):
        """calculates buyers utility"""


    def classification(self):
        """is this agent a prosumer of a consumer? So leader or follower"""
        if self.PvGeneration < self.Consumption:
            self.Classification = ["Leader"]
        else:
            self.Classification = ["Follower"]

    def __repr__(self):
        return " BatteryCapacity:%d PvGeneration:%d, Consumption:%d" % (self.BatteryCapacity, self.PvGeneration, self.Consumption)


class MicroGrid(Model):
    """create environment in which agents can operate"""
    def __init__(self, N):
        self.num_households = N
        self.schedule = RandomActivation(self)
        for e in range(self.num_households):    # create a set of N agents with activations schedule and
            agent = HouseholdAgent(e, household_char.iloc[e, 1], household_char.iloc[e, 2], household_char.iloc[e, 3], self)
            self.schedule.add(agent)

    def step(self):
        """Environment proceeds a step after all agents took a step."""
        self.schedule.step()



