import sys
import time
import os

from source.initialization import *
sys.path.append('/Users/dirkvandenbiggelaar/Desktop/Thesis_workspace/')


""""""""""""""""""""""""
""" MODE SELECTION   """
""""""""""""""""""""""""

""" Solution Scheme  """
""""""""""""""""""""""""
# mode = 'normal'
mode = 'batchrunner'

model = 'sync'
# model = 'async'
#model = 'pso'
#model = 'pso_hierarchical'
#model = 'pso_not_hierarchical'
#model = 'microgrid_no_trading'


if model == 'sync':
    folder_batchrun = 'Batchrun_Sync'
if model == 'async':
    folder_batchrun = 'Batchrun_Async'
if model == 'pso':
    folder_batchrun = 'Batchrun_PSO'
if model == 'pso_hierarchical':
    folder_batchrun = 'Batchrun_PSO_Hierarchy'
if model == 'pso_not_hierarchical':
    folder_batchrun = 'Batchrun_PSONoHierarchy'
if model == 'microgrid_no_trading':
    folder_batchrun = 'Batchrun_NoTrade'


""""""""""""""""""""""""
""" INIT             """
""""""""""""""""""""""""

""" init for normal mode   """
""""""""""""""""""""""""""""""
normal_batchrunner_N = 50
normal_batchrunner_comm_radius = 3

""" Plots  """
""""""""""""""
#plots = 'on'
plots = 'off'

""" init for batchrunner   """
""""""""""""""""""""""""""""""
agents_low = 6
agents_high = 44

every_ = 4

""" general init """
""""""""""""""""""""
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
if model == 'microgrid_no_trading':
    from source.microgrid_no_trading import *

from blockchain.smartcontract import *
from source.plots_normalrunner import *
import os

list_mean_iterations_batch = np.zeros((len(range(agents_low, agents_high)), 3))

def run_mg(sim_steps, N, model_run_mg, args_run_mg):

    np.seterr(all='warn')

    if model_run_mg == 'async':
        [comm_reach_run_mg, lambda_set_run_mg, parameter_sweep_dict_run_mg] = args_run_mg
        number_of_lagging_agents = comm_reach_run_mg
    else:
        [lambda_set_run_mg, parameter_sweep_dict_run_mg] = args_run_mg
        comm_reach_run_mg = None

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

    print('PRE-PROCESSING DATA')

    usable, length_usable = get_usable()
    if length_usable < N:
        pass
        # sys.exit("Number of useable datasets is smaller than number of agents")

    """ Reading in load data from system """
    # sudoPassword = 'biggelaar'
    # command = 'sudo find /Users/dirkvandenbiggelaar/Desktop/DATA/LOAD -name ".DS_Store" -depth -exec rm {} \;'
    # os.system('echo %s|sudo -S %s' % (sudoPassword, command))

    agent_id_load = 0
    number_of_files = 0
    day = 0
    for data_file in os.listdir("/Users/dirkvandenbiggelaar/Desktop/DATA/LOAD"):
        load_file_path = "/Users/dirkvandenbiggelaar/Desktop/DATA/LOAD/" + data_file
        data_return_load= read_csv_load(load_file_path, step_day)
        load_file_agents[agent_id_load][day] = data_return_load
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
            data_return_production = read_csv_production(production_file_path, step_day)
            production_file_agents[agent_id_prod][day] = data_return_production
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


    """ step size scaling of production and consumption """
    for agent in range(N):
        for step in range(int(len(load_file_agents[agent])/step_time)):
            load_file_agents_time[agent][step] = sum(load_file_agents[agent][step_time*step:(step_time*step + step_time)])/step_time
            production_file_agents_time[agent][step] = sum(production_file_agents[agent][step_time*step:(step_time*step + step_time)])/step_time

            load_file_agents_time_med[agent][step] = np.median(load_file_agents[agent][step_time * step:(step_time * step + step_time)]) / step_time
            production_file_agents_time_med[agent][step] = np.median(production_file_agents[agent][step_time * step:(step_time * step + step_time)]) / step_time

    agent_char_load = np.ones(N)
    agent_char_prod = np.ones(N)

    """ random diversity of consumption and production"""
    for i in range(N):
        agent_char_load[i] = random.uniform(1, 3)
        agent_char_prod[i] = random.uniform(1, 2)

    """ Number of prosumers and consumers """
    number_prosumers = int(N * penetration_prosumers)
    number_consumers = N - number_prosumers
    print('number of prosumers:', number_prosumers)
    print('number of consumers:', N - number_prosumers)


    """ First part of agent list consists of prosumers, the rest are consumers"""
    for i in range(number_consumers):
        agent_char_prod[i + number_prosumers] = 0


    total_production = 0
    total_load = 0
    for step in range(sim_steps):
        for agent in range(N):
            total_load += load_file_agents_time[agent][step]**0.5 * sine_wave_consumption_series[step]*agent_char_load[agent]
            total_production += production_file_agents_time[agent][step] * agent_char_prod[agent]


    print('total_production', total_production)
    print('total_load', total_load)

    factor =  total_load / total_production
    number_of_prosumers = np.count_nonzero(agent_char_prod)
    avg_production_sim = total_production / number_of_prosumers
    avg_consumption_sim = total_load/N
    avg_production_day = avg_production_sim / 5
    avg_consumption_day = avg_consumption_sim / 5

    print('average production of 1 prosumer yearly:', avg_production_day)
    print('average consumption of 1 consumer yearly:', avg_consumption_day)

    # if community_size == 'large':
    #     """ extend number of agents to 100 """
    big_data_file = np.zeros((int(total_steps / step_time), N, 3))
    print("COMMUNITY SIZE EXTENDED")
    N_max = 44
    for agent in range(N):
        for step in range(sim_steps):
            if N <= N_max:
                big_data_file[step][agent][0] = load_file_agents_time[agent][step]**0.5 * sine_wave_consumption_series[step]*agent_char_load[agent] /1
                big_data_file[step][agent][1] = production_file_agents_time[agent][step] * agent_char_prod[agent] * factor * 0.95 /1
            if N > N_max:
                # if community_size == 'large':
                agent_copy = random.randint(0, N_max)
                load_factor = random.uniform(0.9, 1.1)
                gen_factor = random.uniform(0.9, 1.1)
                big_data_file[step][agent][0] = big_data_file[step][agent_copy][0] * load_factor
                big_data_file[step][agent][1] = big_data_file[step][agent_copy][1] * gen_factor



                            # """ give a disturbance in load/generation
            #     distrubance of generation on a cloudy day: does prediction work???"""
            # if step > 300 and step < 450:
            #     big_data_file[step][agent][1] = 0



    # if community_size == 'large':
    #     N = N_large
    print('COMMUNITY SIZE: %d' % N)


    load, production, load_series_total, production_series_total = plot_input_data(big_data_file, sim_steps, N)

    print('corrected total_production', production)
    print('corrected total_load', load)
    """Model creation"""
    if model == 'sync':
        model_run = MicroGrid_sync(big_data_file, starting_point, N, lambda_set_run_mg)

    if model == 'async':
        model_run = MicroGrid_async(big_data_file, starting_point, N, number_of_lagging_agents, lambda_set_run_mg)

    if model == 'pso':
        model_run = MicroGrid_PSO(big_data_file, starting_point, N)

    if model == 'pso_hierarchical':
        model_run = MicroGrid_PSO_Hierarchical(big_data_file, starting_point, N, lambda_set_run_mg)

    if model == 'pso_not_hierarchical':
        model_run = MicroGrid_PSO_non_Hierarchical(big_data_file, starting_point, N, lambda_set_run_mg)

    if model == 'microgrid_no_trading':
        model_run = MicroGrid_sync_not_trading(big_data_file, starting_point, N, lambda_set_run_mg)

    if model != 'async':
        number_of_lagging_agents = None

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
    w_sharing_factors_list_over_time = np.zeros((N,sim_steps))

    E_allocated_list_over_time = np.zeros((N,sim_steps))
    revenue_list_over_time = np.zeros((N,sim_steps))
    payment_list_over_time = np.zeros((N,sim_steps))

    """ simple PSO lists """
    results_over_time = np.zeros((N,sim_steps))
    P_supply_list_over_time = np.zeros((N,sim_steps))
    P_demand_list_over_time = np.zeros((N,sim_steps))
    gen_output_list_over_time = np.zeros((N,sim_steps))
    load_demand_list_over_time = np.zeros((N,sim_steps))
    battery_soc_list_over_time = np.zeros((N,sim_steps))
    """Run that fucker"""
    if model == 'pso':
        for i in range(sim_steps):
            """ Data collection PSO Micro-grid"""
            results, P_supply_list, P_demand_list, \
            gen_output_list, load_demand_list, battery_soc_list \
                = model_run.step(N, lambda_set_run_mg)

            for agent in range(N):
                results_over_time[agent][i] = results[agent]
                P_supply_list_over_time[agent][i] = P_supply_list[agent]
                P_demand_list_over_time[agent][i] = P_demand_list[agent]
                gen_output_list_over_time[agent][i] = gen_output_list[agent]
                load_demand_list_over_time[agent][i] = load_demand_list[agent]
                battery_soc_list_over_time[agent][i] = battery_soc_list[agent]
            if i >= stopping_point / step_time:
                print("DATA SAVING")
                break

        print("BATCHRUNNER: BATCH COMPLETE!")


    else:
        for i in range(sim_steps):
            """ Data collection Micro-grid"""
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
                    = model_run.step(N, lambda_set_run_mg)

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
                """ Buyers """
                E_demand_list_over_time[agent][i] = E_total_demand_list[agent]
                E_allocated_list_over_time[agent][i] = E_allocated_list[agent]
                c_prices_over_time[agent][i] = c_nominal_list[agent]
                payment_list_over_time[agent][i] = payment_list[agent]
                """ Sellers"""
                E_surplus_over_time[agent][i] = E_surplus_list[agent]
                E_total_supply_list_over_time[agent][i] = E_total_supply_list[agent]
                E_actual_supplied_list_over_time[agent][i] = E_actual_supplied_list[agent]
                w_sharing_factors_list_over_time[agent][i] = sharing_factors[agent]
                revenue_list_over_time[agent][i] = revenue_list[agent]
                """ Batteries"""
                actual_batteries_over_time[agent][i] = actual_batteries[agent]
                socs_preferred_over_time[agent][i] = soc_preferred_list[agent]
                profit_list_over_time[agent][i] = profit_list[agent]

            if i >= stopping_point/step_time:
                print("DATA SAVING")
                break

        num_global_iteration_over_time_mod = np.delete(num_global_iteration_over_time, [index for index, value in enumerate(num_global_iteration_over_time) if value == 0])
        num_buyer_iteration_over_time_mod = np.delete(num_buyer_iteration_over_time, [index for index, value in enumerate(num_buyer_iteration_over_time) if value == 0])
        num_seller_iteration_over_time_mod = np.delete(num_seller_iteration_over_time, [index for index, value in enumerate(num_seller_iteration_over_time) if value == 0])
        global_mean = np.mean(num_global_iteration_over_time_mod)
        buyer_mean = np.mean(num_buyer_iteration_over_time_mod)
        seller_mean = np.mean(num_seller_iteration_over_time_mod)

        if mode == 'batchrunner':
            np.savez('/Users/dirkvandenbiggelaar/Desktop/result_files/' + folder_batchrun + '/' + str(model) + '_batch' + str(N) + '_laggingagents' + str(number_of_lagging_agents) + '.npz',
                     profit_list_summed_over_time, number_of_buyers_over_time, number_of_sellers_over_time,
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
                     revenue_list_over_time, actual_batteries_over_time, socs_preferred_over_time, profit_list_over_time, w_sharing_factors_list_over_time,
                     big_data_file, sim_steps, R_real_over_time, production_series_total, deficit_total_progress_over_time, N, number_prosumers)

        if mode == 'normal':
            np.savez('/Users/dirkvandenbiggelaar/Desktop/result_files/' + str(model) + '_' + str(N) + '_laggingagents' + str(number_of_lagging_agents) + '.npz',
                     profit_list_summed_over_time, number_of_buyers_over_time, number_of_sellers_over_time,
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
                     revenue_list_over_time, actual_batteries_over_time, socs_preferred_over_time, profit_list_over_time,
                     w_sharing_factors_list_over_time,
                     big_data_file, sim_steps, R_real_over_time, production_series_total, deficit_total_progress_over_time,N, number_prosumers)

        print("DATA SAVED")

        """DATA PROCESSING"""
        if plots == 'on':
            """ execute plots within batchrunner for quick result"""
            close_all()
            plot_profits(profit_list_over_time, profit_list_summed_over_time, N)
            plot_iterations(num_global_iteration_over_time, num_buyer_iteration_over_time,num_seller_iteration_over_time)
            plot_costs_over_time(E_demand_list_over_time, E_allocated_list_over_time, payment_list_over_time, E_total_supply_list_over_time, E_actual_supplied_list_over_time, revenue_list_over_time, N, sim_steps)
            plot_supply_demand(E_total_supply_over_time, E_actual_supplied_total, E_demand_over_time, N)
            plot_w_nominal_progression(w_nominal_over_time, R_prediction_over_time, E_prediction_over_time, E_total_supply_over_time, R_real_over_time, c_nominal_over_time)
            plot_results(w_sharing_factors_list_over_time, E_actual_supplied_list_over_time, E_demand_list_over_time, c_prices_over_time, number_of_buyers_over_time, number_of_sellers_over_time, sim_steps)
            plot_available_vs_supplied(actual_batteries_over_time, E_total_supply_over_time, E_demand_over_time, N)
            plot_utilities(utilities_buyers_over_time, utilities_sellers_over_time, N, sim_steps)
            plot_supplied_vs_surplus_total(surplus_on_step_over_time, E_total_supply_over_time, E_demand_over_time)
            plot_input_data(big_data_file, sim_steps, N)
            plot_avg_soc_preferred(actual_batteries_over_time, socs_preferred_over_time, avg_soc_preferred_over_time, actual_batteries_over_time, deficit_total_over_time, deficit_total_progress_over_time, production_series_total, N, sim_steps, number_prosumers)
            plot_utility_buyer(utilities_buyers_over_time, c_prices_over_time, E_demand_list_over_time, E_surplus_over_time, E_total_supply_over_time, c_nominal_over_time, N, sim_steps)

        length_sim  = len(supplied_over_time_list)

        print("BATCHRUNNER: BATCH COMPLETE!")

        return global_mean, buyer_mean, seller_mean, length_sim

""" Run normal"""
args = [lambda_set, parameter_sweep_dict]
if mode == 'normal':
    N = normal_batchrunner_N
    comm_radius = normal_batchrunner_comm_radius
    if model == 'sync':
        comm_radius = None
        run_mg(sim_steps, N, model, args)
    elif model == 'async':
        """ adaptive communication topology (depending on size of grid)"""
        for comm_reach in range(comm_radius):
            args = [comm_reach, lambda_set, parameter_sweep_dict]
            global_mean, buyer_mean, seller_mean, length_sim = run_mg(sim_steps, N, model, args)
    elif model == 'pso':
        comm_reach = None
        run_mg(sim_steps, N, model, args)
    elif model == 'pso_hierarchical':
        comm_reach = None
        run_mg(sim_steps, N, model, args)
    elif model == 'pso_not_hierarchical':
        comm_reach = None
        run_mg(sim_steps, N, model, args)
    elif model == 'microgrid_no_trading':
        run_mg(sim_steps, N, model, args)


""" Run in batchrunner """
Elapsed_time_batch = []
args = [lambda_set, parameter_sweep_dict]
if mode == 'batchrunner':
    for num_agents in range(agents_low, agents_high):
        comm_radius_low = int((num_agents - 1) / 4)
        comm_radius_high = int((num_agents - 1) / 2)
        if num_agents % every_ == 0:
            tic()
            """ make an simulation for every X population size"""
            if model == 'sync':
                radius = None
                global_mean, buyer_mean, seller_mean, length_sim = run_mg(sim_steps, num_agents, model, args)
            elif model == 'async':
                """ adaptive communication topology (depending on size of grid)"""
                for comm_reach in range(comm_radius_low, comm_radius_high):
                    args = [comm_reach, lambda_set, parameter_sweep_dict]
                    global_mean, buyer_mean, seller_mean, length_sim = run_mg(sim_steps, num_agents, model, args)
            elif model == 'pso':
                comm_reach = None
                global_mean, buyer_mean, seller_mean, length_sim = run_mg(sim_steps, num_agents, model, args)
            elif model == 'pso_hierarchical':
                comm_reach = None
                global_mean, buyer_mean, seller_mean, length_sim = run_mg(sim_steps, num_agents, model, args)
            elif model == 'pso_not_hierarchical':
                comm_reach = None
                global_mean, buyer_mean, seller_mean, length_sim = run_mg(sim_steps, num_agents, model, args)
            elif model == 'microgrid_no_trading':
                global_mean, buyer_mean, seller_mean, length_sim = run_mg(sim_steps, num_agents, model, args)
            Elapsed_time = [toc(), num_agents]
            print(Elapsed_time)
            list_mean_iterations_batch[num_agents - agents_low][:] = [global_mean, buyer_mean, seller_mean]

            Elapsed_time_batch = np.append(Elapsed_time_batch, Elapsed_time)
    np.save('/Users/dirkvandenbiggelaar/Desktop/result_files/'+ folder_batchrun + '/list_mean_iterations_batch', list_mean_iterations_batch)
    np.save('/Users/dirkvandenbiggelaar/Desktop/result_files/'+ folder_batchrun + '/batch_id', [model, agents_low, agents_high, length_sim])
    np.save('/Users/dirkvandenbiggelaar/Desktop/result_files/'+ folder_batchrun + '/Elapsed_time_batch', Elapsed_time_batch)


print('BATCHRUNNER: FULL PROCESS COMPLETE')






