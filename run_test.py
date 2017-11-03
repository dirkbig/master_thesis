from microgrid_model import *
import matplotlib.pyplot as plt
import numpy as np
print(batt1)
time = len(load_data)
N = 3  # N agents only









# Endless amount of empty initial np.array s
step_list = np.array([])                    # open up a list for counting steps
agent_consumption = np.array([])            # consumption pattern per agents
agent_production = np.array([])
agent_capacity = np.array([])


load_over_agents_on_timestep = np.array([])
gen_over_agents_on_timestep = np.array([])
soc_over_agents_on_timestep = np.array([])

load_list_agents_over_time = np.array([])
gen_list_agents_over_time = np.array([])
soc_list_agents_over_time = np.array([])

j = 0

char_list = ["load", "gen", "batt"]

""" this function combines MG data with agent capabilities: e.g. load during the day * unique agent char"""
def calculate_load(char, i, j):
    if char == "load":
        res_fun = load_data[i] * agent_consumption[j]
    elif char == "gen":
        res_fun = rad_data[i] * agent_production[j]
    elif char == "batt":
        res_fun = batt1.get_soc(batt1_out[i])  # what is the soc?
    return res_fun


""" Create Simulation MicroGrid """
model_testrun = MicroGrid(N)  # make microgrid model with N agents
t_scheduled_agents = np.transpose(model_testrun.schedule.agents)  # transpose the schedule list
# print(model_testrun.schedule.agents)


"""Create a list of the characteristics of all agents, foundation: parameters non-changing over time"""
for agent in model_testrun.schedule.agents:
    agent_consumption = np.append(agent_consumption, agent.Consumption)   # for all agents their respective consumption
    agent_production = np.append(agent_production, agent.PvGeneration)    # for all agents their respective production
    agent_capacity = np.append(agent_capacity, agent.BatteryCapacity)     # for all agents their respective battery
    # print(agent_consumption)

t_agent_consumption = np.transpose(agent_consumption)                     # transpose the schedule list

"""ties microgrid data to agents, though non-realistically"""
for i in range(time):   # now takes only 4 steps
    if i > 100:
        break
    model_testrun.step()                                        # make an initial step
    load_over_time_per_agent = np.array([])                     # opens a row for load for 1 agent over time
    for j in range(N):                                          # for every step, review all agents
        agent_consumption_at_timestep = calculate_load(char_list[0],i, j)  # returns the consumption of agent j at time i as a scalar
        agent_production_at_timestep = calculate_load(char_list[1], i, j)
        agent_capacity_at_timestep = calculate_load(char_list[2], i, j)
        load_over_agents_on_timestep = np.append(load_over_agents_on_timestep, agent_consumption_at_timestep)
        gen_over_agents_on_timestep = np.append(gen_over_agents_on_timestep, agent_production_at_timestep)
        soc_over_agents_on_timestep = np.append(soc_over_agents_on_timestep, agent_capacity_at_timestep)
    load_list_agents_over_time = np.append(load_list_agents_over_time, load_over_agents_on_timestep)  # over every timestep
    gen_list_agents_over_time = np.append(gen_list_agents_over_time, gen_over_agents_on_timestep)
    soc_list_agents_over_time = np.append(soc_list_agents_over_time, soc_over_agents_on_timestep)
    load_over_agents_on_timestep = []
    gen_over_agents_on_timestep = []
    soc_over_agents_on_timestep = []


a = np.zeros((len(load_data),len(model_testrun.schedule.agents)))

for i in range(time):   # now takes only i < n steps
    if i > time:
        break
    model_testrun.step()                                        # make an initial step
    load_over_time_per_agent = np.array([])                     # opens a row for load for 1 agent over time
    for j in range(len(model_testrun.schedule.agents)):
        a[i][j] = [calculate_load(char_list[0], i, j), calculate_load(char_list[1], i, j), calculate_load(char_list[2], i, j)]


plt.plot()
plt.show()



# print(len(load_list_agents_over_time))


# plt.plot(load_list_agents_over_time)
# plt.plot(gen_list_agents_over_time)
# plt.plot(soc_list_agents_over_time)


# plt.show()


"""for agent in model_testrun.schedule.agents:
    j += 1
    agent_consumption.append(agent.Consumption)
    agent_production.append(agent.PvGeneration)
    agent_capacity.append(agent.BatteryCapacity)
    for i in range(len(load_data)):
        model_testrun.step()
        step_list.append(i)
        calculate_load(load_data[i], model_testrun.schedule.agents[j-1].Consumption, i, j)
        np.append(agent_consumption, consumption_agent_at_step)


np.append(agent_behaviour, load_agent_at_step)       # add all 3 parameters to agent

print(load_agent1)

# print(model_testrun.schedule.agents)
# print(model_testrun.schedule.agents[0].PvGeneration) # This is how you access object attributes

# print(np.transpose(load_agent1))
# print(np.transpose(model_testrun.schedule.agents))  # This is how an array is transposed


## PLOT ##


plt.plot(load1)
plt.show()"""

print(agent_list[2])

