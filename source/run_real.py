from source.microgrid_model import *

"""Read in actual data specific to actual agent: this is OK (using opensource data)"""


"""Read in real test-data: this is pretty shitty"""
# """Assign data files"""
# load = "load_test.csv"           # load has comma
# production = "solar_test.csv"    # solar has semicolon
#
#
# load_file_agents = np.zeros((N,duration))
# production_file_agents = np.zeros((N,duration))
# # production_file = np.zeros(N)
# battery_file_agents = np.zeros((N,duration))
# master_file = np.zeros((N, duration, 3))
#
#
# """Loads in data of a typical agent"""
# for agent in range(N):
#     load_file_agents[agent] = read_csv(load, duration)
#     production_file_agents[agent] = read_csv(production, duration)
#     battery_file_agents[agent] = np.ones(duration)
#
#
# """Gives all agents initial load and production prediction for the day"""
# big_data_file = np.zeros((duration,N,3))             # list of data_file entries per agents
# for i in range(duration):
#     agent_file = np.zeros((N, 3))  # agent_file
#     for j in range(N):
#         big_data_file[i][j][0] = load_file_agents[j][i]*(random.uniform(0.9, 1.2))
#         big_data_file[i][j][1] = production_file_agents[j][i]*(random.uniform(0.9, 1.2))
#         big_data_file[i][j][2] = battery_file_agents[j][i]
#    # big_data_file[i][0][1] = 0          # for player 1, makes him a consumer
#


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

"""Fake test data: this is complete shit"""
test_load_file_agents = [[100, 100, 100, 100,100, 100, 100, 100,100,100],[70, 100, 100, 100,100, 100, 100, 100,100,100],[0,0,0,0,0,0,0,0,0,0],[50, 100, 100, 100,100, 100, 100, 100,100,100],[0,0,0,0,0,0,0,0,0,0],[100, 100, 100, 100,100, 100, 100, 100,100,100],[70, 100, 100, 100,100, 100, 100, 100,100,100],[0,0,0,0,0,0,0,0,0,0],[50, 100, 100, 100,100, 100, 100, 100,100,100],[0,0,0,0,0,0,0,0,0,0]]
test_production_file_agents = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0],[150,300,150,200,150,150,150,150,150,150],[0,0,0,0,0,0,0,0,0,0],[150,150,150,150,150,150,150,150,150,150],[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0],[150,150,150,150,150,150,150,150,150,150],[0,0,0,0,0,0,0,0,0,0],[150,150,150,150,150,150,150,150,150,150]]
test_battery_file_agents = [100, 100, 100, 100,100, 100, 100, 100,100,100]

big_data_file = np.zeros((duration, N, 3))             # list of data_file entries per agents
for i in range(duration):
    agent_file = np.zeros((N, 3))  # agent_file
    for j in range(N):
        big_data_file[i][j][0] = test_load_file_agents[j][i]            # *(random.uniform(0.9, 1.2))
        big_data_file[i][j][1] = test_production_file_agents[j][i]      # *(random.uniform(0.9, 1.2))
        big_data_file[i][j][2] = test_battery_file_agents[i]

"""Model creation"""
model_testrun = MicroGrid(N, big_data_file)        # create microgrid model with N agents

"""Microgrid ABM makes steps over the duration of the simulation"""
duration_test = duration
supply_over_time_list = []
for i in range(duration):
    supply_per_step, buyers, sellers = model_testrun.step()
    print(supply_per_step)
    supply_over_time_list.append(supply_per_step)

supply_over_time = np.array(supply_over_time_list)
print(supply_over_time)

