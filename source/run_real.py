from source.microgrid_model import *
import os
import matplotlib.pyplot as plt

duration = 1440

"""Read in actual data specific to actual agent: this is OK (using open-source data)"""
load_file_agents = np.zeros((N,duration))
production_file_agents = np.zeros((N,duration))

master_file = np.zeros((N, duration, 3))

agent_id_load = 0

for data_folder in os.listdir("/Users/dirkvandenbiggelaar/Desktop/DATA/LOAD"):
      load_file_path = "/Users/dirkvandenbiggelaar/Desktop/DATA/LOAD/" + data_folder
      load_file_agents[agent_id_load] = read_csv_load(load_file_path, duration)
      agent_id_load += 1
      if agent_id_load > N - 1:
          break

# plt.plot(load_file_agents[0])
# plt.plot(load_file_agents[2])
# #
# plt.show()

usable, length_usable = get_usable()
agent_id_prod = 0


for i in range(len(usable)):                # i is the agent id, this loop loops over number of solar-agents
    production_folder_path = "/Users/dirkvandenbiggelaar/Desktop/DATA/PRODUCTION/" + str(int(usable[i]))
    for file in os.listdir(production_folder_path):
        """take first file only"""
        production_file_path = production_folder_path + '/' + os.listdir(production_folder_path)[1]
        print("agent %d will use %s as its production file" % (i, production_file_path))
        production_file_agents[agent_id_prod] = read_csv_production(production_file_path, duration)
        break
    agent_id_prod += 1
    if agent_id_prod == N:
        break



"""Show production data"""
# for i in range(N):
#     plt.plot(production_file_agents[i])
#     plt.plot(load_file_agents[i]/50)
#     plt.show()

"""Analysis"""
averages_production = np.zeros((N,2))
averages_load = np.zeros((N,2))
for i in range(N):
    """production"""
    averages_production[i][0] = np.sum(production_file_agents[i]) / duration

    averages_production[i][1] = np.amax(production_file_agents[i])
    """load"""
    averages_load[i][0] = np.sum(load_file_agents[i]) / duration
    averages_load[i][1] = np.amax(load_file_agents[i])

"""Read in real test-data: this is pretty shitty"""
#
# """Assign data files"""
# load = "load_test.csv"           # load has comma
# production = "solar_test.csv"    # solar has semicolon
#
#
#"""Loads in data of a typical agent"""
# for agent in range(N):
#     load_file_agents[agent] = read_csv(load, duration)
#     production_file_agents[agent] = read_csv(production, duration)
#     battery_file_agents[agent] = np.ones(duration)
print(averages_production[1:10][0])


plt.plot(averages_production[:][0])
plt.plot(averages_production[:][1])

plt.show()

"""Test with test data: this is almost complete shit"""
# """test - data for a typical agent"""
# test_load = ["test_datafiles.test_load_2_csv.csv","/Users/dirkvandenbiggelaar/Desktop/Thesis_workspace/test_data/test_noload_3_csv.csv"]
# test_production = ["/Users/dirkvandenbiggelaar/Desktop/Thesis_workspace/test_data/test_production_1_csv.csv", "/Users/dirkvandenbiggelaar/Desktop/Thesis_workspace/test_data/test_production_2_csv.csv","/Users/dirkvandenbiggelaar/Desktop/Thesis_workspace/test_data/test_noproduction_3_csv.csv"]
# test_battery = "/Users/dirkvandenbiggelaar/Desktop/Thesis_workspace/test_data/test_battery_forall_csv.csv"

# test_load_file_agents = np.zeros((N,duration))
# test_production_file_agents = np.zeros((N,duration))
# test_battery_file_agents = np.zeros((N,duration))
# # master_file = np.zeros((N, duration, 3))

# for testagent in range(N):
#     test_load_file_agents[testagent] = test_load_file_agents[testagent]         # read_csv([testagent], duration)
#     test_production_file_agents[testagent] = test_production_file_agents[testagent]            # read_csv(test_production[testagent], duration)
#     test_battery_file_agents[testagent] = test_battery_file_agents         # read_csv(test_battery, duration) # all the same battery ""

"""Fake test data: this is actual complete shit"""
# test_load_file_agents =     [ \
#                             [100, 300   ,100, 100,100, 100, 100, 100,100,100],
#                             [100, 100   ,100, 100,100, 100, 100, 100,100,100],
#                             [0  , 0     ,0,0,0,0,0,0,0,0],
#                             [1000, 100   ,100, 100,100, 100, 100, 100,100,100],
#                             [0  , 0     ,0,0,0,0,0,0,0,0],
#                             [100, 100   ,100, 100,100, 100, 100, 100,100,100],
#                             [70 , 100   ,100, 100,100, 100, 100, 100,100,100],
#                             [0  , 0     ,0,0,0,0,0,0,0,0],
#                             [50 , 100   ,100, 100,100, 100, 100, 100,100,100],
#                             [0  , 0     ,0,0,0,0,0,0,0,0]]
#
# test_production_file_agents =  [ \
#                              [0  ,  30  ,0,0,0,0,0,0,0,0],
#                              [0  ,  0    ,0,0,0,0,0,0,0,0],
#                              [20,  310  ,150,200,150,150,150,150,150,150],
#                              [0  ,  0    ,0,0,0,0,0,0,0,0],
#                              [150,  330   ,150,150,150,150,150,150,150,150],
#                              [0  ,  0    ,0,0,0,0,0,0,0,0],
#                              [0  ,  0    ,0,0,0,0,0,0,0,0],
#                              [150,  50   ,150,150,150,150,150,150,150,150],
#                              [0  ,  0    ,0,0,0,0,0,0,0,0],
#                              [150,  150  ,150,150,150,150,150,150,150,150]]
#
# test_battery_file_agents = [100, 100, 100, 100,100, 100, 100, 100,100,100]
#
# big_data_file = np.zeros((duration, N, 3))             # list of data_file entries per agents
# for i in range(duration):
#     agent_file = np.zeros((N, 3))  # agent_file
#     for j in range(N):
#         big_data_file[i][j][0] = test_load_file_agents[j][i]            # *(random.uniform(0.9, 1.2))
#         big_data_file[i][j][1] = test_production_file_agents[j][i]      # *(random.uniform(0.9, 1.2))
#         big_data_file[i][j][2] = test_battery_file_agents[i]







"""Assigns all agents initial load and production prediction for the day"""
big_data_file = np.zeros((N, duration, 3))             # list of data_file entries per agents
for step in range(duration):
    agent_file = np.zeros((N, 3))  # agent_file
    for agent in range(N):
        big_data_file[agent][step][0] = load_file_agents[agent][step] # load_file_agents[j][i]*(random.uniform(0.9, 1.2))
        big_data_file[agent][step][1] = production_file_agents[agent][step]  #  production_file_agents[j][i]*(random.uniform(0.9, 1.2))
        # big_data_file[agent][step][2] = battery_file_agents[agent][step]
   # big_data_file[i][0][1] = 0          # for player 1, makes him a consumer

"""Model creation"""
model_testrun = MicroGrid(N, big_data_file)        # create microgrid model with N agents

"""Microgrid ABM makes steps over the duration of the simulation"""
duration_test = duration
supply_over_time_list = []

"""Run that fucker"""
for i in range(duration):
    surplus_on_step, supply_on_step, buyers, sellers = model_testrun.step()
    print("total surplus =", surplus_on_step, "supplied =", supply_on_step)
    supply_over_time_list.append(supply_on_step)

supply_over_time = np.array(supply_over_time_list)

