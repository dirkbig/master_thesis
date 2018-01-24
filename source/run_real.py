from source.microgrid_model import *
import os

if stopping_point > total_steps:
    sys.exit("stopping point should be within bounds of the day")

""" Prelim"""


Fs = sim_steps
f = 20
sample = sim_steps
sine_wave_consumption_series = np.zeros(sim_steps)
sine_constant = 1
for i in range(sim_steps):
    sine_wave_consumption_series[i] = sine_constant + 0.7 * np.sin(np.pi * f * i / Fs - 1 * np.pi)


"""Read in actual data specific to actual agent: this is OK (using open-source data)"""
load_file_agents = np.zeros((N, days, step_day))
production_file_agents = np.zeros((N, days, step_day))
master_file = np.zeros((N, step_day, 3))
agent_id_load = 0
number_of_files = 0
day = 0

for data_folder in os.listdir("/Users/dirkvandenbiggelaar/Desktop/DATA/LOAD"):
    load_file_path = "/Users/dirkvandenbiggelaar/Desktop/DATA/LOAD/" + data_folder
    load_file_agents[agent_id_load][day] = read_csv_load(load_file_path, step_day)
    number_of_files += 1
    agent_id_load += 1
    if number_of_files % N == 0:
        day += 1
        agent_id_load = 0
    if number_of_files > (N*days - 1):
        break

usable, length_usable = get_usable()
if length_usable < N:
    sys.exit("Number of useable datasets is smaller than number of agents")


agent_id_prod = 0
number_of_files = 0
day = 0

for i in range(len(usable)):                # i is the agent id, this loop loops over number of solar-agents
    production_folder_path = "/Users/dirkvandenbiggelaar/Desktop/DATA/PRODUCTION/" + str(int(usable[i]))
    for file in os.listdir(production_folder_path):
        """take first file only"""
        production_file_path = production_folder_path + '/' + file
        # print("agent %d will use %s as its production file" % (i, production_file_path))
        production_for_the_day = read_csv_production(production_file_path, step_day)
        production_file_agents[agent_id_prod][day] = production_for_the_day
        number_of_files += 1
        day += 1
        if number_of_files % days == 0:
            break
    day = 0
    agent_id_prod += 1
    if number_of_files > (N*days - 1):
        break



"""Normalise data before appending them together"""
#
# for i in range(N):
#     for day in range(days):
#         for step in range(step_day):
#             if load_file_agents[i][day][step] > 30*np.mean(load_file_agents[i][day]):
#                 load_file_agents[i][day][step] = np.mean(load_file_agents[i][day])
#             if
#         load_file_agents[i][day] = load_file_agents[i][day] /(np.mean(load_file_agents[i][day]))
#         production_file_agents[i][day] = production_file_agents[i][day]/np.mean(production_file_agents[i][day])

for i in range(N):
         load_file_agents[i] = load_file_agents[i] / np.mean(load_file_agents[i])
#         production_file_agents[i] = production_file_agents[i]/np.mean(production_file_agents[i])

load_file_agents.resize((N, total_steps))
production_file_agents.resize((N, total_steps))


# """The long list of data per step"""
# load_file_agents_days = np.zeros((N,step_day*days))
# production_file_agents_days = np.zeros((N,step_day*days))
#
# load_file_agents_extended = []
# production_file_agents_extended = []
# for i in range(N):
#     for day in range(days):
#         for step in range(step_day):
#             load_file_agents_extended.append([load_file_agents_extended,load_file_agents[i][day][step]])
#             production_file_agents_extended.append([production_file_agents_extended, production_file_agents[i][day][step]])
#     # print(load_file_agents_appended.shape)
#     # print(load_file_agents_appended.shape)
#     load_file_agents_extended = np.asarray(load_file_agents_extended)
#     production_file_agents_extended = np.asarray(production_file_agents_extended)
#
#     load_file_agents_days[i] = load_file_agents_extended
#     production_file_agents_days[i] = production_file_agents_extended
#     print(load_file_agents_days.shape)
#     print(production_file_agents_days.shape)
#
#
# for i in range(N):
#     plt.plot(load_file_agents[i][day])
#     # plt.plot(load_file_agents_days[i])
#     plt.show()
#



# for i in range(N):
#     plt.plot(load_file_agents_days[i])
#     # plt.plot(production_file_agents[i])
#     plt.show()


load_file_agents.resize((N, total_steps))
production_file_agents.resize((N, total_steps))


"""Data Analysis"""
avg_max_production = np.zeros((N,2))
avg_max_load = np.zeros((N,2))
for i in range(N):
    avg_max_production[i][0] = np.sum(production_file_agents[i]) / total_steps
    avg_max_production[i][1] = np.amax(production_file_agents[i])
    avg_max_load[i][0] = np.sum(load_file_agents[i]) / total_steps
    avg_max_load[i][1] = np.amax(load_file_agents[i])


"""Read in real test-data: this is pretty shitty"""
#
# """Assign data files"""
# load = "load_test.csv"           # load has comma
# production = "solar_test.csv"    # solar has semicolon
#
#
#"""Loads in data of a typical agent"""
# for agent in range(N):
#     load_file_agents[agent] = read_csv(load, total_step)
#     production_file_agents[agent] = read_csv(production, total_step)
#     battery_file_agents[agent] = np.ones(total_step)

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






#
#
"""write to a big_data_files"""
# for i in range(N):
#     with open('/Users/dirkvandenbiggelaar/Desktop/DATA/big_data_file_load/data_load_agent' + str(int(i)) + '.csv', 'w', newline='') as data_load_agent:
#         writer = csv.writer(data_load_agent)
#         for j in range(len(load_file_agents[i])):
#             writer.writerow([load_file_agents[i][j]])
#     with open('/Users/dirkvandenbiggelaar/Desktop/DATA/big_data_file_production/data_production_agent' + str(int(i)) + '.csv', 'w', newline='') as data_production_agent:
#         writer = csv.writer(data_production_agent)
#         for j in range(len(production_file_agents[i])):
#             writer.writerow([production_file_agents[i][j]])
#
#
# # load_file_agents (N,total_step)
# # production_file_agents
#
#
# load_file_agents_loaded = np.zeros((N, total_steps))
# production_file_agents_loaded = np.zeros((N, total_steps))
#
# """read in big_data_file"""
# agent_id_load = 0
#
# for file_data in os.listdir("/Users/dirkvandenbiggelaar/Desktop/DATA/big_data_file_load"):
#     print()
#     load_file_path = "/Users/dirkvandenbiggelaar/Desktop/DATA/big_data_file_load/" + file_data
#     load_file_agents_loaded[agent_id_load] = read_csv_big_data(load_file_path, total_steps)
#     agent_id_load += 1
#
# agent_id_load = 0
#
# plt.plot(load_file_agents_loaded[1])
# plt.show()


"""Assigns all agents initial load and production prediction for the day"""
sim_steps = int(total_steps/step_time)
big_data_file = np.zeros((int(total_steps/step_time), N, 3))             # list of data_file entries per agents
#

load_file_agents_time = np.zeros((N, int(total_steps/step_time)))
production_file_agents_time = np.zeros((N, int(total_steps/step_time)))

load_file_agents_time_med = np.zeros((N, int(total_steps/step_time)))
production_file_agents_time_med = np.zeros((N, int(total_steps/step_time)))


for agent in range(N):
    for step in range(int(len(load_file_agents[agent])/step_time)):
        load_file_agents_time[agent][step] = sum(load_file_agents[agent][step_time*step:(step_time*step + step_time)])/step_time
        production_file_agents_time[agent][step] = sum(production_file_agents[agent][step_time*step:(step_time*step + step_time)])/step_time
        load_file_agents_time_med[agent][step] = np.median(load_file_agents[agent][step_time * step:(step_time * step + step_time)]) / step_time
        production_file_agents_time_med[agent][step] = np.median(production_file_agents[agent][step_time * step:(step_time * step + step_time)]) / step_time

for step in range(sim_steps):
    for agent in range(N):
        big_data_file[step][agent][0] = load_file_agents_time[agent][step]**0.5 * sine_wave_consumption_series[step]
        big_data_file[step][agent][1] = production_file_agents_time[agent][step]


load, production, load_series_total, production_series_total = plot_input_data(big_data_file,sim_steps, N)

"""Model creation"""
model_testrun = MicroGrid(N, big_data_file, starting_point)        # create microgrid model with N agents

"""Microgrid ABM makes steps over the duration of the simulation, data collection"""
duration_test = sim_steps
supplied_over_time_list = np.zeros(sim_steps)
mean_sharing_factors = np.zeros(sim_steps)
demand_over_time = np.zeros(sim_steps)
c_nominal_over_time = np.zeros(sim_steps)
number_of_buyers_over_time = np.zeros(sim_steps)
number_of_sellers_over_time = np.zeros(sim_steps)
w_nominal_over_time = np.zeros(sim_steps)
R_prediction_over_time = np.zeros(sim_steps)
E_prediction_over_time = np.zeros(sim_steps)
R_real_over_time = np.zeros(sim_steps)
E_real_over_time = np.zeros(sim_steps)
utilities_buyers_over_time = np.zeros((sim_steps, N, 4))
utilities_sellers_over_time = np.zeros((sim_steps, N, 3))

surplus_on_step_over_time = np.zeros(sim_steps)
supplied_on_step_over_time = np.zeros(sim_steps)
demand_on_step_over_time = np.zeros(sim_steps)

""" batteries """
actual_batteries_over_time = np.zeros((N, sim_steps))
E_total_supply_over_time = np.zeros(sim_steps)
E_demand_over_time = np.zeros(sim_steps)
avg_soc_preferred_over_time = np.zeros(sim_steps)
soc_preferred_list_over_time = np.zeros((N, sim_steps))

plot_C_P(load_series_total, production_series_total)

"""Run that fucker"""
for i in range(sim_steps):
    """ Data collection"""
    surplus_on_step, supplied_on_step, demand_on_step, \
    buyers, sellers, sharing_factors, \
    c_nominal_per_step, w_nominal, \
    R_prediction_step, E_prediction_step, E_real, R_real, \
    actual_batteries, E_total_supply, E_demand, \
    utilities_buyers, utilities_sellers, \
    soc_preferred_list, avg_soc_preferred \
        = model_testrun.step()

    # E_real == supplied_on_step

    mean_sharing_factors[i] = np.mean(sharing_factors)
    supplied_over_time_list[i] = supplied_on_step
    demand_over_time[i] = demand_on_step
    c_nominal_over_time[i] = c_nominal_per_step

    """ Fix this """
    number_of_buyers_over_time[i] = len(buyers)
    number_of_sellers_over_time[i] = len(sellers)

    surplus_on_step_over_time[i] = surplus_on_step
    supplied_on_step_over_time[i] = supplied_on_step
    demand_on_step_over_time[i] = demand_on_step


    R_prediction_over_time[i] = R_prediction_step
    E_prediction_over_time[i] = E_prediction_step
    w_nominal_over_time[i] = w_nominal
    R_real_over_time[i] = R_real
    E_real_over_time[i] = E_real

    for agent in range(N):
        actual_batteries_over_time[agent][i] = actual_batteries[agent]

    utilities_sellers_over_time[i][:][:] = utilities_sellers
    utilities_buyers_over_time[i][:][:] = utilities_buyers

    E_total_supply_over_time[i] = E_total_supply
    E_demand_over_time[i] = E_demand

    # print(utilities_sellers, 'utilities_sellers')
    # print(utilities_sellers_over_time, 'utilities_sellers_over_time')

    if i >= stopping_point/step_time:
        print("done, nu nog plotjes")
        break




"""DATA PROCESSING, oftewel plots"""

plot_w_nominal_progression(w_nominal_over_time, R_prediction_over_time, E_prediction_over_time, E_real_over_time, R_real_over_time, c_nominal_over_time)
plot_results(mean_sharing_factors, supplied_over_time_list, demand_over_time, c_nominal_over_time,number_of_buyers_over_time,number_of_sellers_over_time)
plot_available_vs_supplied(actual_batteries_over_time, E_total_supply_over_time, E_demand_over_time, N)
plot_utilities(utilities_buyers_over_time, utilities_sellers_over_time, N, sim_steps)
plot_supplied_vs_surplus_total(surplus_on_step_over_time, supplied_on_step_over_time, demand_on_step_over_time)


plot_input_data(big_data_file, sim_steps, N)
# plot_avg_soc_preferred(soc_preferred_list_over_time, avg_soc_preferred_over_time)

print("done, nu echt")



""" Run validation """

for i in range(sim_steps):
    pass
