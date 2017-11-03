from microgrid_model import *
from function_file import *
import numpy as np



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


"""Loads in data"""
load_file = read_csv(load, duration)
production_file = read_csv(production, duration)
battery_file = np.ones(duration)
master_file = [load_file, production_file]

"""gives all agents initial load and production prediction of the day"""
agent_file = np.zeros((duration, 3))
list_of_agents = np.zeros(N)
for i in range(N):                                                       # "for agent in model_testrun.schedule.agents:"
    for j in range(duration):
        agent_file[j][0] = load_file[j]
        agent_file[j][1] = production_file[j]
        agent_file[j][2] = battery_file[j]
    list_of_agents = np.append(list_of_agents, agent_file)


# MODEL CREATION
model_testrun = MicroGrid(N)        # create microgrid model with N agents


# print(agent_file)
# print(list_of_agents[1])

#
#
# Simulation

"""Microgrid ABM makes steps over the duration of the simulation"""
duration_test = duration
for i in range(duration):
    model_testrun.step()
    for j in range(N):
        print(model_testrun.schedule.agents[j].PvGeneration)


#
#
# TESTING
"""
print(model_testrun.schedule.agents)
print(model_testrun.schedule.agents[1].Consumption)


"""

# plt.plot(load_file)
# plt.plot(agent_file[1])
# plt.show()


