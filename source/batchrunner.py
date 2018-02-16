import sys
from source.initialization import *
sys.path.append('/Users/dirkvandenbiggelaar/Desktop/Thesis_workspace/')


mode = 'normal'
# mode = 'batchrunner'


#model = 'pso'
#model = 'sync'
model = 'async'

agents_low = 5
agents_high = 20

lambda11 = 2
lambda12 = 1
lambda21 = 2
lambda22 = 2
lambda_set = [lambda11, lambda12, lambda21, lambda22]


if model == 'sync':
    from source.microgrid_model import *
if model == 'async':
    from source.microgrid_async import *
if model == 'pso':
    from source.PSO_model import *

from blockchain.smartcontract import *
from source.plots import *
import os


""" Default init settings"""

def run_mg(sim_steps, N, comm_reach, lambda_set):

    np.seterr(all='warn')

    if stopping_point > total_steps:
        sys.exit("stopping point should be within bounds of the day")

    """ Prelim """
    Fs = sim_steps
    f = 20
    sample = sim_steps
    sine_wave_consumption_series = np.zeros(sim_steps)
    sine_constant = 3
    for i in range(sim_steps):
        sine_wave_consumption_series[i] = sine_constant + 0.7 * np.sin(np.pi * f * i / Fs - 1 * np.pi)

    """Read in actual data specific to actual agent: this is OK (using open-source data)"""
    load_file_agents = np.zeros((N, days, step_day))
    production_file_agents = np.zeros((N, days, step_day))
    master_file = np.zeros((N, step_day, 3))


    usable, length_usable = get_usable()
    if length_usable < N:
        sys.exit("Number of useable datasets is smaller than number of agents")

    """ Reading in load data from system """
    # sudoPassword = 'biggelaar'
    # command = 'sudo find /Users/dirkvandenbiggelaar/Desktop/DATA/LOAD -name ".DS_Store" -depth -exec rm {} \;'
    # os.system('echo %s|sudo -S %s' % (sudoPassword, command))

    agent_id_load = 0
    number_of_files = 0
    day = 0
    for data_file in os.listdir("/Users/dirkvandenbiggelaar/Desktop/DATA/LOAD"):
        if number_of_files == 69:
            pass
        load_file_path = "/Users/dirkvandenbiggelaar/Desktop/DATA/LOAD/" + data_file
        load_file_agents[agent_id_load][day] = read_csv_load(load_file_path, step_day)
        number_of_files += 1
        agent_id_load += 1
        if number_of_files % N == 0:
            day += 1
            agent_id_load = 0
        if number_of_files > (N*days - 1):
            break

    # sudoPassword = 'biggelaar'
    # command = 'sudo find /Users/dirkvandenbiggelaar/Desktop/DATA/PRODUCTION -name ".DS_Store" -depth -exec rm {} \;'
    # os.system('echo %s|sudo -S %s' % (sudoPassword, command))

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

    for i in range(N):
             load_file_agents[i] = load_file_agents[i] / np.mean(load_file_agents[i])

    load_file_agents.resize((N, total_steps))
    production_file_agents.resize((N, total_steps))

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


    """Assigns all agents initial load and production prediction for the day"""
    sim_steps = int(total_steps/step_time)
    big_data_file = np.zeros((int(total_steps/step_time), N, 3))


    load_file_agents_time = np.zeros((N, int(total_steps/step_time)))
    production_file_agents_time = np.zeros((N, int(total_steps/step_time)))

    load_file_agents_time_med = np.zeros((N, int(total_steps/step_time)))
    production_file_agents_time_med = np.zeros((N, int(total_steps/step_time)))

    for agent in range(min(N,14)):
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
    if model == 'sync':
        model_run = MicroGrid_sync(big_data_file, starting_point, N, lambda_set)        # create microgrid model with N agents

    if model == 'async':
        model_run = MicroGrid_async(big_data_file, starting_point, N, comm_reach, lambda_set)        # create microgrid model with N agents


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
    socs_preferred_over_time = np.zeros((N, sim_steps))
    E_total_demand_over_time = np.zeros((N, sim_steps))
    c_prices_over_time = np.zeros((N, sim_steps))
    E_surplus_over_time = np.zeros((N, sim_steps))

    num_global_iteration_over_time = np.zeros(sim_steps)
    num_buyer_iteration_over_time = np.zeros(sim_steps)
    num_seller_iteration_over_time = np.zeros(sim_steps)
    profit_list_over_time = np.zeros((N,sim_steps))
    profit_list_summed_over_time = np.zeros(sim_steps)
    """Run that fucker"""
    for i in range(sim_steps):
        """ Data collection"""
        surplus_on_step, supplied_on_step, demand_on_step, \
        buyers, sellers, sharing_factors, \
        c_nominal_per_step, w_nominal, \
        R_prediction_step, E_prediction_step, E_real, R_real, \
        actual_batteries, E_total_supply, E_demand, \
        utilities_buyers, utilities_sellers, \
        soc_preferred_list, avg_soc_preferred, \
        E_total_demand_list, c_nominal_list, E_surplus_list, \
        num_global_iteration, num_buyer_iteration, num_seller_iteration,\
        profit_list \
                = model_run.step(N, lambda_set)

        # E_real == supplied_on_step
        profit_list_summed_over_time[i] = sum(profit_list)
        mean_sharing_factors[i] = np.mean(sharing_factors)
        supplied_over_time_list[i] = supplied_on_step
        demand_over_time[i] = demand_on_step
        c_nominal_over_time[i] = c_nominal_per_step
        avg_soc_preferred_over_time[i] = avg_soc_preferred
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
            E_total_demand_over_time[agent][i] = E_total_demand_list[agent]
            c_prices_over_time[agent][i] = c_nominal_list[agent]
            E_surplus_over_time[agent][i] = E_surplus_list[agent]
            socs_preferred_over_time[agent][i] = soc_preferred_list[agent]
            profit_list_over_time[agent][i] = profit_list[agent]
        utilities_sellers_over_time[i][:][:] = utilities_sellers
        utilities_buyers_over_time[i][:][:] = utilities_buyers

        E_surplus_over_time
        E_total_supply_over_time[i] = E_total_supply
        E_demand_over_time[i] = E_demand
        # print(utilities_sellers, 'utilities_sellers')
        # print(utilities_sellers_over_time, 'utilities_sellers_over_time')
        num_global_iteration_over_time[i] = num_global_iteration
        num_buyer_iteration_over_time[i] = num_buyer_iteration
        num_seller_iteration_over_time[i] = num_seller_iteration
        if i >= stopping_point/step_time:
            print("done, nu nog plotjes")
            break

    plot_profits(profit_list_over_time, profit_list_summed_over_time, N)
    plot_iterations(num_global_iteration_over_time, num_buyer_iteration_over_time,num_seller_iteration_over_time)
    num_global_iteration_over_time = np.delete(num_global_iteration_over_time, [index for index, value in enumerate(num_global_iteration_over_time) if value == 0])
    num_buyer_iteration_over_time = np.delete(num_buyer_iteration_over_time, [index for index, value in enumerate(num_buyer_iteration_over_time) if value == 0])
    num_seller_iteration_over_time = np.delete(num_seller_iteration_over_time, [index for index, value in enumerate(num_seller_iteration_over_time) if value == 0])
    global_mean = np.mean(num_global_iteration_over_time)
    buyer_mean = np.mean(num_buyer_iteration_over_time)
    seller_mean = np.mean(num_seller_iteration_over_time)

    close_all()
    """DATA PROCESSING, oftewel plots"""
    plot_w_nominal_progression(w_nominal_over_time, R_prediction_over_time, E_prediction_over_time, E_real_over_time, R_real_over_time, c_nominal_over_time)
    plot_results(mean_sharing_factors, supplied_over_time_list, demand_over_time, c_nominal_over_time,number_of_buyers_over_time,number_of_sellers_over_time)
    plot_available_vs_supplied(actual_batteries_over_time, E_total_supply_over_time, E_demand_over_time, N)
    plot_utilities(utilities_buyers_over_time, utilities_sellers_over_time, N, sim_steps)
    plot_supplied_vs_surplus_total(surplus_on_step_over_time, supplied_on_step_over_time, demand_on_step_over_time)
    plot_input_data(big_data_file, sim_steps, N)
    plot_avg_soc_preferred(socs_preferred_over_time, avg_soc_preferred_over_time, actual_batteries_over_time, N, sim_steps)
    plot_utility_buyer(utilities_buyers_over_time, c_prices_over_time, E_total_demand_over_time, E_surplus_over_time, E_total_supply, c_nominal_over_time, N, sim_steps)
    # plot_utility_seller(utilities_sellers_over_time, w_factors_over_time, E_total_demand_over_time, w_nominal_over_time, N, sim_steps)

    print("done, nu echt")

    return global_mean, buyer_mean, seller_mean

def run_mg_pso(sim_steps, N, comm_reach, lambda_set):

    np.seterr(all='warn')

    if stopping_point > total_steps:
        sys.exit("stopping point should be within bounds of the day")

    """ Prelim """
    Fs = sim_steps
    f = 20
    sample = sim_steps
    sine_wave_consumption_series = np.zeros(sim_steps)
    sine_constant = 3
    for i in range(sim_steps):
        sine_wave_consumption_series[i] = sine_constant + 0.7 * np.sin(np.pi * f * i / Fs - 1 * np.pi)

    """Read in actual data specific to actual agent: this is OK (using open-source data)"""
    load_file_agents = np.zeros((N, days, step_day))
    production_file_agents = np.zeros((N, days, step_day))
    master_file = np.zeros((N, step_day, 3))


    usable, length_usable = get_usable()
    if length_usable < N:
        sys.exit("Number of useable datasets is smaller than number of agents")

    """ Reading in load data from system """
    # sudoPassword = 'biggelaar'
    # command = 'sudo find /Users/dirkvandenbiggelaar/Desktop/DATA/LOAD -name ".DS_Store" -depth -exec rm {} \;'
    # os.system('echo %s|sudo -S %s' % (sudoPassword, command))

    agent_id_load = 0
    number_of_files = 0
    day = 0
    for data_file in os.listdir("/Users/dirkvandenbiggelaar/Desktop/DATA/LOAD"):
        if number_of_files == 69:
            pass
        load_file_path = "/Users/dirkvandenbiggelaar/Desktop/DATA/LOAD/" + data_file
        load_file_agents[agent_id_load][day] = read_csv_load(load_file_path, step_day)
        number_of_files += 1
        agent_id_load += 1
        if number_of_files % N == 0:
            day += 1
            agent_id_load = 0
        if number_of_files > (N*days - 1):
            break


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

    for i in range(N):
             load_file_agents[i] = load_file_agents[i] / np.mean(load_file_agents[i])

    load_file_agents.resize((N, total_steps))
    production_file_agents.resize((N, total_steps))

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

    """Assigns all agents initial load and production prediction for the day"""
    sim_steps = int(total_steps/step_time)
    big_data_file = np.zeros((int(total_steps/step_time), N, 3))


    load_file_agents_time = np.zeros((N, int(total_steps/step_time)))
    production_file_agents_time = np.zeros((N, int(total_steps/step_time)))

    load_file_agents_time_med = np.zeros((N, int(total_steps/step_time)))
    production_file_agents_time_med = np.zeros((N, int(total_steps/step_time)))

    for agent in range(min(N,14)):
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
    if model == 'sync':
        model_run = MicroGrid_sync(big_data_file, starting_point, N, lambda_set)        # create microgrid model with N agents

    if model == 'async':
        model_run = MicroGrid_async(big_data_file, starting_point, N, comm_reach, lambda_set)        # create microgrid model with N agents

    if model == 'pso':
        PSO_run = MicroGrid_PSO(big_data_file, starting_point, N)

    results_over_time = np.zeros((N, sim_steps))
    P_supply_list_over_time = np.zeros(sim_steps)
    P_demand_list_over_time = np.zeros(sim_steps)
    gen_output_list_over_time = np.zeros(sim_steps)
    load_demand_list_over_time = np.zeros(sim_steps)
    battery_soc_list_over_time = np.zeros(sim_steps)
    for i in range(sim_steps):
        results, P_supply_list, P_demand_list, gen_output_list, load_demand_list,  battery_soc_list = PSO_run.pso_step(big_data_file, N)

        for agent in range(N):
            results_over_time[agent][i] = results[agent]

        P_supply_list_over_time[i] = sum(P_supply_list)
        P_demand_list_over_time[i] = sum(P_demand_list)
        gen_output_list_over_time[i] =  sum(gen_output_list)
        load_demand_list_over_time[i] =  sum(load_demand_list)
        battery_soc_list_over_time[i]  =  np.mean(battery_soc_list)

    close_all()
    plot_PSO(results_over_time, P_supply_list_over_time, P_demand_list_over_time, gen_output_list_over_time, load_demand_list_over_time, battery_soc_list_over_time, N, sim_steps)

    """ PSO results"""

list_mean_iterations_batch = np.zeros((len(range(agents_low, agents_high)), 3))

""" Run normal"""
if mode == 'normal':
    N = 14
    comm_radius = 3
    if model == 'sync':
        comm_radius = None
        run_mg(sim_steps, N, comm_radius, lambda_set)
    elif model == 'async':
        run_mg(sim_steps, N, comm_radius, lambda_set)
    elif model == 'pso':
        comm_reach = None
        run_mg_pso(sim_steps, N, comm_reach, lambda_set)

""" Run in batchrunner """
if mode == 'batchrunner':
    for num_agents in range(agents_low, agents_high):
        if model == 'sync':
            radius = None
            global_mean, buyer_mean, seller_mean = run_mg(sim_steps, num_agents, radius, lambda_set)
            print(global_mean, buyer_mean, seller_mean)
        elif model == 'async':
            """ adaptive communication topology (depending on size of grid)"""
            comm_radius_low = int((num_agents-1)/4)
            comm_radius_high = int((num_agents-1)/2)
            for radius in range(comm_radius_low, comm_radius_high):
                global_mean, buyer_mean, seller_mean = run_mg(sim_steps, num_agents, radius)
                print(global_mean, buyer_mean, seller_mean)
        elif model == 'pso':
            comm_reach = None
            run_mg_pso(sim_steps, N, comm_reach, lambda_set)

        list_mean_iterations_batch[num_agents - agents_low][:] = global_mean, buyer_mean, seller_mean
        np.save('/Users/dirkvandenbiggelaar/Desktop/python_plots/list_mean_iterations_batch', list_mean_iterations_batch)

print(list_mean_iterations_batch)

list_mean_iterations_batch_loaded = np.load('/Users/dirkvandenbiggelaar/Desktop/python_plots/list_mean_iterations_batch.npy')

