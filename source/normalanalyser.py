import numpy as np
from source.plots_batchrunner import *
from source.plots_normalrunner import *

close_all()

batchdata_zip = np.load('/Users/dirkvandenbiggelaar/Desktop/python_plots/batchdata/sync_batch25_commreachisNone.npz')
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
                      'number_of_agents',
                      'number_prosumers'), (batchdata_zip[batch] for batch in batchdata_zip)))

profit_list_summed_over_time = batchdata['profit_list_summed_over_time']
number_of_buyers_over_time = batchdata['number_of_buyers_over_time']
number_of_sellers_over_time = batchdata['number_of_sellers_over_time']
supplied_over_time_list = batchdata['supplied_over_time_list']
E_demand_over_time = batchdata['E_demand_over_time']
c_nominal_over_time = batchdata['c_nominal_over_time']
deficit_total_over_time = batchdata['deficit_total_over_time']
avg_soc_preferred_over_time = batchdata['avg_soc_preferred_over_time']
surplus_on_step_over_time = batchdata['surplus_on_step_over_time']
w_nominal_over_time = batchdata['w_nominal_over_time']
mean_sharing_factors = batchdata['mean_sharing_factors']
w_sharing_factors_list_over_time = batchdata['w_sharing_factors_list_over_time']
E_prediction_over_time = batchdata['E_prediction_over_time']
R_prediction_over_time = batchdata['R_prediction_over_time']
E_total_supply_over_time = batchdata['E_total_supply_over_time']
num_global_iteration_over_time = batchdata['num_global_iteration_over_time']
num_buyer_iteration_over_time = batchdata['num_buyer_iteration_over_time']
num_seller_iteration_over_time = batchdata['num_seller_iteration_over_time']
utilities_sellers_over_time = batchdata['utilities_sellers_over_time']
utilities_buyers_over_time = batchdata['utilities_buyers_over_time']
E_consumption_list_over_time = batchdata['E_consumption_list_over_time']
E_production_list_over_time = batchdata['E_production_list_over_time']
E_demand_list_over_time = batchdata['E_demand_list_over_time']
E_allocated_list_over_time = batchdata['E_allocated_list_over_time']
c_prices_over_time = batchdata['c_prices_over_time']
payment_list_over_time = batchdata['payment_list_over_time']
E_surplus_over_time = batchdata['E_surplus_over_time']
E_total_supply_list_over_time = batchdata['E_total_supply_list_over_time']
E_actual_supplied_list_over_time = batchdata['E_actual_supplied_list_over_time']
revenue_list_over_time = batchdata['revenue_list_over_time']
actual_batteries_over_time = batchdata['actual_batteries_over_time']
socs_preferred_over_time = batchdata['socs_preferred_over_time']
profit_list_over_time = batchdata['profit_list_over_time']
R_real_over_time = batchdata['R_real_over_time']
deficit_total_progress_over_time = batchdata['deficit_total_progress_over_time']
production_series_total = batchdata['production_series_total']
number_prosumers = batchdata['number_prosumers']


N = batchdata['number_of_agents']
big_data_file = batchdata['big_data_file']
sim_steps = batchdata['sim_steps']

plot_profits(profit_list_over_time, profit_list_summed_over_time, N)
plot_iterations(num_global_iteration_over_time, num_buyer_iteration_over_time,num_seller_iteration_over_time)
plot_costs_over_time(E_demand_list_over_time, E_allocated_list_over_time, payment_list_over_time, E_total_supply_list_over_time, E_actual_supplied_list_over_time, revenue_list_over_time, N, sim_steps)
plot_supply_demand(E_total_supply_over_time, E_actual_supplied_list_over_time, E_demand_over_time, N)
plot_w_nominal_progression(w_nominal_over_time, R_prediction_over_time, E_prediction_over_time, E_total_supply_over_time, R_real_over_time, c_nominal_over_time)
plot_results(w_sharing_factors_list_over_time, E_actual_supplied_list_over_time, E_demand_list_over_time, c_prices_over_time, number_of_buyers_over_time, number_of_sellers_over_time, sim_steps)
plot_available_vs_supplied(actual_batteries_over_time, E_total_supply_over_time, E_demand_over_time, N)
plot_supplied_vs_surplus_total(surplus_on_step_over_time, E_total_supply_over_time, E_demand_over_time)
plot_input_data(big_data_file, sim_steps, N)
plot_avg_soc_preferred(actual_batteries_over_time, socs_preferred_over_time, avg_soc_preferred_over_time, actual_batteries_over_time, deficit_total_over_time, deficit_total_progress_over_time, production_series_total, N, sim_steps, number_prosumers)
plot_utility_buyer(utilities_buyers_over_time, c_prices_over_time, E_demand_list_over_time, E_surplus_over_time, E_total_supply_over_time, c_nominal_over_time, N, sim_steps)

