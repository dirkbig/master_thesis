import sys
import time
from source.initialization import *
sys.path.append('/Users/dirkvandenbiggelaar/Desktop/Thesis_workspace/')

""""""""""""""""""""""""
""" MODE SELECTION   """
""""""""""""""""""""""""
mode = 'normal'
# mode = 'batchrunner'

""" EnergyBazaar """
model = 'sync'
#model = 'async'

""" PSO """
#model = 'pso'
#model = 'pso_hierarchical'
#model = 'pso_not_hierarchical'


plots = 'on'
# plots = 'off'

""""""""""""""""""""""""
""" INIT             """
""""""""""""""""""""""""

penetration_prosumers = 0.5
lambda11 = 2
lambda12 = 1
lambda21 = 2
lambda22 = 2
lambda_set = [lambda11, lambda12, lambda21, lambda22]

range_parameter_sweep = 10
horizon_low = 70
horizon_high = 1440
batt_low = 0
batt_high = 10

parameter_sweep_dict = {'max_horizon': np.arange(horizon_low, horizon_high, range_parameter_sweep),
                        'battery_capacity': np.arange(batt_low, batt_high, range_parameter_sweep)}



""" init for normal mode """
normal_batchrunner_N = 20
normal_batchrunner_comm_radius = 3

""" init for batchrunner mode """
agents_low = 5
agents_high = 44


""""""""""""""""""""""""
""" START            """
""""""""""""""""""""""""

if model == 'sync':
    from source.microgrid_model import *
if model == 'async':
    from source.microgrid_async import *
if model == 'pso':
    from source.PSO_model import *
if model == 'pso_hierarchical':
    from source.microgrid_PSO_hierarchical import *
if model == 'pso_not_hierarchical':
    from source.microgrid_PSO_no_Hierarchy import *

from blockchain.smartcontract import *
from source.plots import *
import os

list_mean_iterations_batch = np.zeros((len(range(agents_low, agents_high)), 3))

def run_mg(sim_steps, N, args):

    np.seterr(all='warn')
    if mode == 'sync':
        [lambda_set, parameter_sweep_dict] = args


    if mode == 'async':
        [comm_reach, lambda_set, parameter_sweep_dict] = args

    if stopping_point > total_steps:
        sys.exit("stopping point should be within bounds of the day")

    """ Prelim """
    Fs = sim_steps
    f = 10
    sample = sim_steps
    sine_wave_consumption_series = np.zeros(sim_steps)
    sine_constant = 1
    for i in range(sim_steps):
        sine_wave_consumption_series[i] =0.5 * (sine_constant + 0.7 * np.sin(np.pi * f * i / Fs + 0.25 * np.pi))

    """Read in actual data specific to actual agent: this is OK (using open-source data)"""
    load_file_agents = np.zeros((N, days, step_day))
    production_file_agents = np.zeros((N, days, step_day))


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


    agent_char_load = np.ones(N)
    agent_char_prod = np.ones(N)

    """ random diversity of consumption"""
    for i in range(N):
        agent_char_load[i] = random.uniform(1, 3)
        agent_char_prod[i] = random.uniform(1, 2)

    """ Number of pure producers and pure consumers """
    pure_producers = 0  # percent
    pure_consumers = 1 - penetration_prosumers  # percent
    number_pure_producers = int(N * pure_producers)
    number_pure_consumers = int(N * pure_consumers)
    print('number of prosumers:', N - number_pure_consumers)
    for i in range(number_pure_producers):
        agent_char_load[i] = 0

    for i in range(number_pure_consumers):
        agent_char_prod[i + number_pure_producers] = 0

    total_production = 0
    total_load = 0
    for step in range(sim_steps):
        for agent in range(N):
            total_load += load_file_agents_time[agent][step]**0.5 * sine_wave_consumption_series[step]*agent_char_load[agent]
            total_production += production_file_agents_time[agent][step] * agent_char_prod[agent]


    print('total_production', total_production)
    print('total_load', total_load)

    factor =  total_load / total_production
    print(factor)
    number_of_prosumers = np.count_nonzero(agent_char_prod)
    avg_production_week = total_production / number_of_prosumers
    avg_production_year = avg_production_week * 52
    avg_consumption_year = total_load/N *52

    print('average production of 1 prosumer yearly:', avg_production_year)
    print('average consumption of 1 consumer yearly:', avg_consumption_year)

    for step in range(sim_steps):
        for agent in range(N):
            big_data_file[step][agent][0] = load_file_agents_time[agent][step]**0.5 * sine_wave_consumption_series[step]*agent_char_load[agent]
            big_data_file[step][agent][1] = production_file_agents_time[agent][step] * agent_char_prod[agent] * factor


    load, production, load_series_total, production_series_total = plot_input_data(big_data_file,sim_steps, N)


    print('corrected total_production', production)
    print('corrected total_load', load)
    """Model creation"""
    if model == 'sync':
        model_run = MicroGrid_sync(big_data_file, starting_point, N, lambda_set)

    if model == 'async':
        model_run = MicroGrid_async(big_data_file, starting_point, N, comm_reach, lambda_set)

    if model == 'pso':
        model_run = MicroGrid_PSO(big_data_file, starting_point, N)

    if model == 'pso_hierarchical':
        model_run = MicroGrid_PSO(big_data_file, starting_point, N, lambda_set)

    if model == 'pso_not_hierarchical':
        model_run = MicroGrid_PSO_non_Hierarchical(big_data_file, starting_point, N, lambda_set)

    """Microgrid ABM makes steps over the duration of the simulation, data collection"""
    E_consumption_list_over_time = np.zeros((N, sim_steps))
    E_production_list_over_time = np.zeros((N, sim_steps))

    supplied_over_time_list = np.zeros(sim_steps)
    mean_sharing_factors = np.zeros(sim_steps)
    c_nominal_over_time = np.zeros(sim_steps)
    number_of_buyers_over_time = np.zeros(sim_steps)
    number_of_sellers_over_time = np.zeros(sim_steps)
    w_nominal_over_time = np.zeros(sim_steps)
    R_prediction_over_time = np.zeros(sim_steps)
    E_prediction_over_time = np.zeros(sim_steps)
    R_real_over_time = np.zeros(sim_steps)
    utilities_buyers_over_time = np.zeros((sim_steps, N, 4))
    utilities_sellers_over_time = np.zeros((sim_steps, N, 3))
    surplus_on_step_over_time = np.zeros(sim_steps)
    """ batteries """
    actual_batteries_over_time = np.zeros((N, sim_steps))
    E_total_supply_over_time = np.zeros(sim_steps)
    E_demand_over_time = np.zeros(sim_steps)
    avg_soc_preferred_over_time = np.zeros(sim_steps)
    socs_preferred_over_time = np.zeros((N, sim_steps))
    E_demand_list_over_time = np.zeros((N, sim_steps))
    c_prices_over_time = np.zeros((N, sim_steps))
    E_surplus_over_time = np.zeros((N, sim_steps))
    E_actual_supplied_total = np.zeros(sim_steps)
    num_global_iteration_over_time = np.zeros(sim_steps)
    num_buyer_iteration_over_time = np.zeros(sim_steps)
    num_seller_iteration_over_time = np.zeros(sim_steps)
    profit_list_over_time = np.zeros((N,sim_steps))
    profit_list_summed_over_time = np.zeros(sim_steps)
    deficit_total_over_time = np.zeros(sim_steps)
    deficit_total_progress_over_time = np.zeros(sim_steps)
    E_total_supply_list_over_time = np.zeros((N,sim_steps))
    E_actual_supplied_list_over_time = np.zeros((N,sim_steps))

    E_allocated_list_over_time = np.zeros((N,sim_steps))
    revenue_list_over_time = np.zeros((N,sim_steps))
    payment_list_over_time = np.zeros((N,sim_steps))

    """Run that fucker"""
    for i in range(sim_steps):
        """ Data collection"""
        E_surplus_total, E_demand_total, \
        buyers, sellers, sharing_factors, \
        c_nominal_per_step, w_nominal, \
        R_prediction_step, E_prediction_step, R_real, \
        actual_batteries, E_total_supply, \
        utilities_buyers, utilities_sellers, \
        soc_preferred_list, avg_soc_preferred, \
        E_consumption_list, E_production_list, \
        E_total_demand_list, c_nominal_list, E_surplus_list, E_total_supply_list,\
        num_global_iteration, num_buyer_iteration, num_seller_iteration,\
        profit_list, revenue_list, payment_list,\
        deficit_total, deficit_total_progress, E_actual_supplied_list, E_allocated_list \
                = model_run.step(N, lambda_set)

        profit_list_summed_over_time[i] = sum(profit_list)
        number_of_buyers_over_time[i] = len(buyers)
        number_of_sellers_over_time[i] = len(sellers)

        supplied_over_time_list[i] = E_total_supply
        E_demand_over_time[i] = E_demand_total
        c_nominal_over_time[i] = c_nominal_per_step

        deficit_total_over_time[i] = deficit_total
        deficit_total_progress_over_time[i] = deficit_total_progress
        avg_soc_preferred_over_time[i] = avg_soc_preferred

        surplus_on_step_over_time[i] = E_surplus_total
        w_nominal_over_time[i] = w_nominal
        mean_sharing_factors[i] = np.mean(sharing_factors)
        E_prediction_over_time[i] = E_prediction_step
        R_prediction_over_time[i] = R_prediction_step
        R_real_over_time[i] = R_real
        E_total_supply_over_time[i] = E_total_supply
        E_actual_supplied_total[i] = sum(E_actual_supplied_list)

        num_global_iteration_over_time[i] = num_global_iteration
        num_buyer_iteration_over_time[i] = num_buyer_iteration
        num_seller_iteration_over_time[i] = num_seller_iteration
        utilities_sellers_over_time[i][:][:] = utilities_sellers
        utilities_buyers_over_time[i][:][:] = utilities_buyers

        """ Lists """
        for agent in range(N):
            E_consumption_list_over_time[agent][i] = E_consumption_list[agent]
            E_production_list_over_time[agent][i] = E_production_list[agent]

            E_demand_list_over_time[agent][i] = E_total_demand_list[agent]
            E_allocated_list_over_time[agent][i] = E_allocated_list[agent]
            c_prices_over_time[agent][i] = c_nominal_list[agent]
            payment_list_over_time[agent][i] = payment_list[agent]

            E_surplus_over_time[agent][i] = E_surplus_list[agent]
            E_total_supply_list_over_time[agent][i] = E_total_supply_list[agent]
            E_actual_supplied_list_over_time[agent][i] = E_actual_supplied_list[agent]
            revenue_list_over_time[agent][i] = revenue_list[agent]

            actual_batteries_over_time[agent][i] = actual_batteries[agent]
            socs_preferred_over_time[agent][i] = soc_preferred_list[agent]
            profit_list_over_time[agent][i] = profit_list[agent]

        if i >= stopping_point/step_time:
            print("done, nu nog data processing")
            break

    # np.save('/Users/dirkvandenbiggelaar/Desktop/python_plots/batch/N',
    #         N)
    # np.save('/Users/dirkvandenbiggelaar/Desktop/python_plots/batch/E_demand_list_over_time',
    #         E_demand_list_over_time)
    # np.save('/Users/dirkvandenbiggelaar/Desktop/python_plots/batch/E_allocated_list_over_time',
    #         E_allocated_list_over_time)
    # np.save('/Users/dirkvandenbiggelaar/Desktop/python_plots/batch/payment_list_over_time',
    #         payment_list_over_time)
    # np.save('/Users/dirkvandenbiggelaar/Desktop/python_plots/batch/E_total_supply_list_over_time',
    #         E_total_supply_list_over_time)
    # np.save('/Users/dirkvandenbiggelaar/Desktop/python_plots/batch/E_actual_supplied_list_over_time',
    #         E_actual_supplied_list_over_time)
    # np.save('/Users/dirkvandenbiggelaar/Desktop/python_plots/batch/revenue_list_over_time',
    #         revenue_list_over_time)
    # np.save('/Users/dirkvandenbiggelaar/Desktop/python_plots/batch/list_mean_iterations_batch',
    #         mean_sharing_factors)
    # np.save('/Users/dirkvandenbiggelaar/Desktop/python_plots/batch/list_mean_iterations_batch',
    #         c_nominal_over_time)
    # np.save('/Users/dirkvandenbiggelaar/Desktop/python_plots/batch/list_mean_iterations_batch',
    #         number_of_buyers_over_time)
    # np.save('/Users/dirkvandenbiggelaar/Desktop/python_plots/batch/list_mean_iterations_batch',
    #         number_of_sellers_over_time)
    # np.save('/Users/dirkvandenbiggelaar/Desktop/python_plots/batch/list_mean_iterations_batch',
    #         list_mean_iterations_batch)


    num_global_iteration_over_time_mod = np.delete(num_global_iteration_over_time, [index for index, value in enumerate(num_global_iteration_over_time) if value == 0])
    num_buyer_iteration_over_time_mod = np.delete(num_buyer_iteration_over_time, [index for index, value in enumerate(num_buyer_iteration_over_time) if value == 0])
    num_seller_iteration_over_time_mod = np.delete(num_seller_iteration_over_time, [index for index, value in enumerate(num_seller_iteration_over_time) if value == 0])
    global_mean = np.mean(num_global_iteration_over_time_mod)
    buyer_mean = np.mean(num_buyer_iteration_over_time_mod)
    seller_mean = np.mean(num_seller_iteration_over_time_mod)



    """DATA PROCESSING, oftewel plots"""
    if plots == 'on':
        close_all()

        plot_profits(profit_list_over_time, profit_list_summed_over_time, N)
        plot_iterations(num_global_iteration_over_time, num_buyer_iteration_over_time,num_seller_iteration_over_time)

        plot_costs_over_time(E_demand_list_over_time, E_allocated_list_over_time, payment_list_over_time, E_total_supply_list_over_time, E_actual_supplied_list_over_time, revenue_list_over_time, N, sim_steps)
        plot_supply_demand(E_total_supply_over_time, E_actual_supplied_total, E_demand_over_time, N)
        plot_w_nominal_progression(w_nominal_over_time, R_prediction_over_time, E_prediction_over_time, E_total_supply_over_time, R_real_over_time, c_nominal_over_time)
        plot_results(mean_sharing_factors, supplied_over_time_list, E_demand_over_time, c_nominal_over_time, number_of_buyers_over_time, number_of_sellers_over_time)
        plot_available_vs_supplied(actual_batteries_over_time, E_total_supply_over_time, E_demand_over_time, N)
        plot_utilities(utilities_buyers_over_time, utilities_sellers_over_time, N, sim_steps)
        plot_supplied_vs_surplus_total(surplus_on_step_over_time, E_total_supply_over_time, E_demand_over_time)
        plot_input_data(big_data_file, sim_steps, N)
        plot_avg_soc_preferred(actual_batteries_over_time, socs_preferred_over_time, avg_soc_preferred_over_time, actual_batteries_over_time, deficit_total_over_time, deficit_total_progress_over_time, production_series_total, N, sim_steps)
        plot_utility_buyer(utilities_buyers_over_time, c_prices_over_time, E_demand_list_over_time, E_surplus_over_time, E_total_supply_over_time, c_nominal_over_time, N, sim_steps)


    length_sim  = len(supplied_over_time_list)
    np.savez('/Users/dirkvandenbiggelaar/Desktop/python_plots/batchdata/batch_data_Nis' + str(N) + '_commreachis' + str(comm_reach) + '.npz', profit_list_summed_over_time, number_of_buyers_over_time, number_of_sellers_over_time,
             supplied_over_time_list, E_demand_over_time,
             c_nominal_over_time, deficit_total_over_time, deficit_total_progress_over_time,
             avg_soc_preferred_over_time, surplus_on_step_over_time, w_nominal_over_time,
             mean_sharing_factors, E_prediction_over_time, R_prediction_over_time, R_real_over_time,
             E_total_supply_over_time, E_actual_supplied_total, num_global_iteration_over_time,
             num_buyer_iteration_over_time, num_seller_iteration_over_time, utilities_sellers_over_time,
             utilities_buyers_over_time,
             E_consumption_list_over_time, E_production_list_over_time, E_demand_list_over_time,
             E_allocated_list_over_time, c_prices_over_time, payment_list_over_time,
             E_surplus_over_time, E_total_supply_list_over_time, E_actual_supplied_list_over_time,
             revenue_list_over_time, actual_batteries_over_time, socs_preferred_over_time, profit_list_over_time, N)

    print("done, nu echt!")

    return global_mean, buyer_mean, seller_mean, length_sim

def run_mg_pso(sim_steps, N, args):

    np.seterr(all='warn')

    if stopping_point > total_steps:
        sys.exit("stopping point should be within bounds of the day")

    """ Prelim """
    Fs = sim_steps
    f = 20
    sine_wave_consumption_series = np.zeros(sim_steps)
    sine_constant = 3
    for i in range(sim_steps):
        sine_wave_consumption_series[i] = sine_constant + 0.7 * np.sin(np.pi * f * i / Fs - 1 * np.pi)

    """Read in actual data specific to actual agent: this is OK (using open-source data)"""
    load_file_agents = np.zeros((N, days, step_day))
    production_file_agents = np.zeros((N, days, step_day))


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

    agent_char_load = np.ones(N)
    agent_char_prod = np.ones(N)

    """ random diversity of consumption"""
    for i in range(N):
        agent_char_load[i] = random.uniform(1, 3)
        agent_char_prod[i] = random.uniform(1, 2)

    """ Number of pure producers and pure consumers """
    pure_producers = 0.2  # percent
    pure_consumers = 1 - penetration_prosumers  # percent
    number_pure_producers = int(N * pure_producers)
    number_pure_consumers = int(N * pure_consumers)
    for i in range(number_pure_producers):
        agent_char_load[i] = 0

    for i in range(number_pure_consumers):
        agent_char_prod[i + number_pure_producers] = 0

    load, production, load_series_total, production_series_total = plot_input_data(big_data_file,sim_steps, N)

    number_of_prosumers = np.count_nonzero(agent_char_prod)
    avg_production_week = production / number_of_prosumers
    avg_production_year = avg_production_week * 52
    avg_consumption_year = load/N *52
    print('total load this week:', load)
    print('total production this week:', production)
    print('average production yearly:', avg_production_year)
    print('average consumption yearly:', avg_consumption_year)

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

""" Run normal"""
args = [lambda_set, parameter_sweep_dict]
if mode == 'normal':
    N = normal_batchrunner_N
    comm_radius = normal_batchrunner_comm_radius
    if model == 'sync':
        comm_radius = None
        run_mg(sim_steps, N, args)
    elif model == 'async':
        args = [comm_reach, lambda_set, parameter_sweep_dict]
        run_mg(sim_steps, N, args)
    elif model == 'pso':
        comm_reach = None
        run_mg_pso(sim_steps, N, args)
    elif model == 'pso_hierarchical':
        comm_reach = None
        run_mg(sim_steps, N, args)
    elif model == 'pso_not_hierarchical':
        comm_reach = None
        run_mg(sim_steps, N, args)

""" Run in batchrunner """
args = [lambda_set, parameter_sweep_dict]
if mode == 'batchrunner':
    for num_agents in range(agents_low, agents_high):
        comm_radius_low = int((num_agents - 1) / 4)
        comm_radius_high = int((num_agents - 1) / 2)
        if model == 'sync':
            radius = None
            global_mean, buyer_mean, seller_mean, length_sim = run_mg(sim_steps, num_agents, args)
        elif model == 'async':
            """ adaptive communication topology (depending on size of grid)"""
            for comm_reach in range(comm_radius_low, comm_radius_high):
                args = [comm_reach, lambda_set, parameter_sweep_dict]
                global_mean, buyer_mean, seller_mean = run_mg(sim_steps, num_agents, args)
        elif model == 'pso':
            comm_reach = None
            run_mg_pso(sim_steps, num_agents, args)
        elif model == 'pso_hierarchical':
            comm_reach = None
            run_mg(sim_steps, num_agents, args)

        list_mean_iterations_batch[num_agents - agents_low][:] = [global_mean, buyer_mean, seller_mean]

    np.save('/Users/dirkvandenbiggelaar/Desktop/python_plots/batchdata/list_mean_iterations_batch', list_mean_iterations_batch)
    np.save('/Users/dirkvandenbiggelaar/Desktop/python_plots/batchdata/batch_id_' + time.strftime(
        "%H_%M_%S") + "__" + time.strftime("%d_%m_%Y"),
            [agents_low, agents_high, length_sim])

print('BATCHRUNNER: PROCESS COMPLETE')






