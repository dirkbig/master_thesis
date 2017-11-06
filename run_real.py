from microgrid_model import *
from function_file import *
import numpy as np
import matplotlib.pyplot as plt



#
#
# INITIALIZATION
"""All starting parameters are initialised"""
duration = 1440                     # Duration of the sim (one full day or 1440 time_steps of 1 minute)
N = 3                               # N agents only
step_list = np.zeros([duration])


"""Assign data files"""
load = "load_test.csv"           # load has comma
production = "solar_test.csv"    # solar has semicolon


load_file_agents = np.zeros((N,duration))
production_file_agents = np.zeros((N,duration))
# production_file = np.zeros(N)
battery_file_agents = np.zeros((N,duration))
master_file = np.zeros((N, duration, 3))


"""Loads in data of a typical agent"""
for agent in range(N):
    load_file_agents[agent] = read_csv(load, duration)
    production_file_agents[agent] = read_csv(production, duration)
    battery_file_agents[agent] = np.ones(duration)


"""Gives all agents initial load and production prediction for the day"""
big_data_file = np.zeros((duration,N,3))             # list of data_file entries per agents
for i in range(duration):
    agent_file = np.zeros((N, 3))  # agent_file
    for j in range(N):
        big_data_file[i][j][0] = load_file_agents[j][i]
        big_data_file[i][j][1] = production_file_agents[j][i]
        big_data_file[i][j][2] = battery_file_agents[j][i]
    big_data_file[i][0][1] = 0          # for player 1, makes him a consumer


# MODEL CREATION
model_testrun = MicroGrid(N, big_data_file)        # create microgrid model with N agents

# import pdb
# pdb.set_trace()

# print(agent_file)
# print(list_of_agents[1])

#
#
# Simulation

"""Microgrid ABM makes steps over the duration of the simulation"""
duration_test = duration
supply_over_time_list = []
for i in range(duration):
    supply, buyers_pool, sellers_pool = model_testrun.step()
    supply_over_time_list.append(supply)

supply_over_time = np.array(supply_over_time_list)
#
#
# TESTING
"""
print(model_testrun.schedule.agents)
print(model_testrun.schedule.agents[1].Consumption)


"""

# plt.plot(supply_over_time)
# plt.plot(agent_file[1])
# plt.show()


