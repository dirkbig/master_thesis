import numpy as np
from source.plots_batchrunner import *
container = []




batch_folder = 'NoTrade'


elapsed_time = np.load('/Users/dirkvandenbiggelaar/Desktop/result_files/' + batch_folder + '/batch_id.npy')
batch_id = np.load('/Users/dirkvandenbiggelaar/Desktop/result_files/' + batch_folder + '/Elapsed_time_batch.npy')
list_mean_iterations_batch = np.load('/Users/dirkvandenbiggelaar/Desktop/result_files/' + batch_folder + '/list_mean_iterations_batch.npy')

agents_low, agents_high, length_sim = batch_id
num_steps = length_sim
range_agents = agents_high - agents_low
num_batches = len(range(agents_low, agents_high))
comm_reach = None


for N in range(agents_low, agents_high):
    batchdata_zip = np.load('/Users/dirkvandenbiggelaar/Desktop/result_files/' + batch_folder + '/Batchrun_EBZsync_batch' + str(N) + '_commreachis' + str(comm_reach) + '.npz')
    batchdata = dict(zip(('profit_list_summed_over_time',
                      'number_of_buyers_over_time',
                      'number_of_sellers_over_time',
                      'supplied_over_time_list',
                      'E_demand_over_time',
                      'c_nominal_over_time',
                      'deficit_total_over_time',
                      'deficit_total_progress_over_time',
                      'avg_soc_preferred_over_time',
                      'surplus_on_step_over_time',
                      'w_nominal_over_time',
                      'mean_sharing_factors',
                      'E_prediction_over_time',
                      'R_prediction_over_time',
                      'R_real_over_time',
                      'E_total_supply_over_time',
                      'E_actual_supplied_total',
                      'num_global_iteration_over_time',
                      'num_buyer_iteration_over_time',
                      'num_seller_iteration_over_time',
                      'utilities_sellers_over_time',
                      'utilities_buyers_over_time',
                      'E_consumption_list_over_time',
                      'E_production_list_over_time',
                      'E_demand_list_over_time',
                      'E_allocated_list_over_time',
                      'c_prices_over_time',
                      'payment_list_over_time',
                      'E_surplus_over_time',
                      'E_total_supply_list_over_time',
                      'E_actual_supplied_list_over_time',
                      'revenue_list_over_time',
                      'actual_batteries_over_time',
                      'socs_preferred_over_time',
                      'profit_list_over_time',
                      'w_sharing_factors_list_over_time',
                      'big_data_file',
                      'sim_steps',
                      'R_real_over_time',
                      'production_series_total',
                      'deficit_total_progress_over_time',
                      'number_of_agents', 'number_prosumers'), (batchdata_zip[batch] for batch in batchdata_zip)))
    container = np.append(container, batchdata)


print('BATCH ANALYSER: DATA LOADED')

num_batches = agents_high - agents_low
num_buyer_iteration_over_time_batch = np.zeros((num_batches, num_steps))
num_global_iteration_over_time_batch = np.zeros((num_batches, num_steps))
num_seller_iteration_over_time_batch = np.zeros((num_batches, num_steps))
deficit_total_over_time_batch = np.zeros((num_batches,num_steps))



c_nominal_over_time_batch = np.zeros((num_batches, num_steps))
w_nominal_over_time_batch = np.zeros((num_batches, num_steps))
E_total_supply_list_over_time_batch = []

actual_batteries_list_over_time_batch = []
socs_preferred_over_time_batch = []
E_actual_supplied_total_batch = []

num_total_rows = 0
for batch in range(agents_high - agents_low):
    N = agents_low + batch
    num_total_rows += N
    # deficit_total_over_time[batch] = container[batch]['deficit_total_over_time']
    # deficit_total_progress_over_time[batch] = container[batch]['deficit_total_progress_over_time']
    c_nominal_over_time_batch[batch] = container[batch]['c_nominal_over_time']
    w_nominal_over_time_batch[batch] = container[batch]['w_nominal_over_time']


    E_total_supply_list_over_time = container[batch]['E_total_supply_list_over_time']
    E_total_supply_list_over_time_batch = np.append(E_total_supply_list_over_time_batch,
                                                    [E_total_supply_list_over_time])

    actual_batteries_list_over_time = container[batch]['actual_batteries_over_time']
    actual_batteries_list_over_time_batch = np.append(actual_batteries_list_over_time_batch,
                                                    [actual_batteries_list_over_time])

    socs_preferred_over_time = container[batch]['socs_preferred_over_time']
    socs_preferred_over_time_batch = np.append(socs_preferred_over_time_batch,
                                                    [socs_preferred_over_time])

    E_actual_supplied_total = container[batch]['E_actual_supplied_total']
    E_actual_supplied_total_batch = np.append(E_actual_supplied_total_batch,
                                                    [E_actual_supplied_total])

    # num_buyer_iteration_over_time_batch[batch] = container[batch]['num_buyer_iteration_over_time']
    # num_seller_iteration_over_time_batch[batch] = container[batch]['num_seller_iteration_over_time']
    # num_global_iteration_over_time_batch[batch] = container[batch]['num_global_iteration_over_time']
    #

    # profit_list_summed_over_time, \
    # number_of_buyers_over_time, \
    # number_of_sellers_over_time, \
    # supplied_over_time_list, \
    # E_demand_over_time,\
    # avg_soc_preferred_over_time, \
    # surplus_on_step_over_time, \
    # w_nominal_over_time,\
    # mean_sharing_factors, \
    # E_prediction_over_time, \
    # R_prediction_over_time, \
    # R_real_over_time,\
    # E_total_supply_over_time, \
    # E_actual_supplied_total, \
    # utilities_sellers_over_time,\
    # utilities_buyers_over_time,\
    # E_consumption_list_over_time, \
    # E_production_list_over_time, \
    # E_demand_list_over_time,\
    # E_allocated_list_over_time, \
    # c_prices_over_time, \
    # payment_list_over_time,\
    # E_surplus_over_time, \
    # E_total_supply_list_over_time, \
    # E_actual_supplied_list_over_time,\
    # revenue_list_over_time, \
    # actual_batteries_over_time, \
    # socs_preferred_over_time, \
    # profit_list_over_time, \
    # number_of_agents = container[batch]

    # for step in range(num_steps):
        # num_global_iteration_over_time_batch[batch][step] = num_global_iteration_over_time[step]
        # num_buyer_iteration_over_time_batch[batch][step]  = num_buyer_iteration_over_time[step]
        # num_seller_iteration_over_time_batch[batch][step]  = num_seller_iteration_over_time[step]

E_total_supply_list_over_time_batch = np.reshape(E_total_supply_list_over_time_batch, (num_total_rows, num_steps))
actual_batteries_list_over_time_batch = np.reshape(actual_batteries_list_over_time_batch, (num_total_rows, num_steps))



E_total_supply_list_over_time_mean = np.zeros((num_batches,num_steps))
N = agents_low
batch_row = 0
for batch in range(num_batches):
    E_total_supply_list_over_time = E_total_supply_list_over_time_batch[batch_row:(batch_row + N)]
    w_nominal_over_time_batch
    for step in range(num_steps):
        E_total_supply_list_over_time_mean[batch][step] = 0
        for agent in range(N):
            E_total_supply_list_over_time_mean[batch][step] += np.mean(E_total_supply_list_over_time[agent][step])

    batch_row += N
    N += + 1


print('BATCH ANALYSER: DATA PROCESSED')


thesis_supply_demand_batch_plot(E_total_supply_list_over_time_mean, w_nominal_over_time_batch, num_batches)
thesis_iteration_plot(list_mean_iterations_batch, num_batches, range_agents, agents_low, agents_high)
thesis_control_values_plot(c_nominal_over_time_batch, w_nominal_over_time_batch, num_batches)
thesis_soc_batch_plot(actual_batteries_list_over_time_batch, socs_preferred_over_time_batch, E_actual_supplied_total_batch, num_batches, agents_low, num_steps)