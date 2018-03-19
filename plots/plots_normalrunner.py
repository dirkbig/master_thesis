import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.dates as md
import datetime as dt

""" kleur TU-DELFT bies: 00A6D6"""
sns.set()
c=sns.color_palette()[0]
b=sns.color_palette()[1]
a=sns.color_palette()[2]
d=sns.color_palette()[3]

fig_width = (13,6)
figsize_single = (13,3)
figsize_double = (13,6)

days = 5

def close_all():
    plt.close("all")

def plot_avg_soc_preferred(actual_batteries_over_time, soc_preferred_list_over_time, avg_soc_preferred_over_time, soc_actual_over_time, deficit_total_over_time, deficit_total_progress_over_time, production_series_total, N, steps, number_prosumers):

    fig_soc_preferred = plt.figure(figsize=figsize_double, dpi=500)
    plt.tight_layout(w_pad=0.2, h_pad=0.4)
    x_steps = np.arange(steps)


    number_prosumers = number_prosumers
    number_consumers = N - number_prosumers
    deficit_total_over_time_avg = abs(deficit_total_over_time) / N
    avg_deficit_total_over_time = deficit_total_over_time / N
    std_soc_preferred_list_over_time = np.std(soc_preferred_list_over_time, axis=0)
    std_actual_batteries_over_time = np.std(actual_batteries_over_time, axis=0)
    min_soc_preferred_list_over_time = np.min(soc_preferred_list_over_time, axis=0)
    max_soc_preferred_list_over_time = np.max(soc_preferred_list_over_time, axis=0)
    min_actual_batteries_over_time = np.min(actual_batteries_over_time, axis=0)
    max_actual_batteries_over_time = np.max(actual_batteries_over_time, axis=0)

    max_battery = max(max_actual_batteries_over_time)
    max_deficit = max(deficit_total_over_time_avg)

    avg_soc_actual_over_time_consumers = np.zeros(steps)
    avg_soc_actual_over_time_prosumers = np.zeros(steps)
    avg_soc_actual_over_time = np.zeros(steps)
    for agent in range(N):
        for i in range(steps):
            avg_soc_actual_over_time[i] += soc_actual_over_time[agent][i]/N
            if agent < number_prosumers:
                avg_soc_actual_over_time_prosumers[i] += soc_actual_over_time[agent][i]/number_prosumers
            elif agent >= number_prosumers:
                avg_soc_actual_over_time_consumers[i] += soc_actual_over_time[agent][i]/number_consumers


    ax1 = fig_soc_preferred.add_subplot(211)
    ax2 = fig_soc_preferred.add_subplot(212)


    ax1.fill_between(x_steps, avg_soc_actual_over_time - std_actual_batteries_over_time, avg_soc_actual_over_time + std_actual_batteries_over_time,color=c, alpha=0.3, label='SD')
    ax1.fill_between(x_steps, min_actual_batteries_over_time, avg_soc_actual_over_time - std_actual_batteries_over_time,color=a, alpha=0.1)
    ax1.fill_between(x_steps, max_actual_batteries_over_time, avg_soc_actual_over_time + std_actual_batteries_over_time,color=a, alpha=0.1, label='min-max')
    ax1.plot(x_steps, avg_soc_actual_over_time_prosumers, color=b, linestyle='-', label='mean soc prosumers', alpha=0.9)
    ax1.plot(x_steps, avg_soc_actual_over_time_consumers, color=d, linestyle='-', label='mean soc consumers', alpha=0.9)

    ax2.plot(x_steps, deficit_total_over_time_avg, color=a, linestyle='--', label='deficit')

    for day in range(days):
        ax1.axvline(steps / days * (day + 1), color='k', linestyle='-', alpha=0.1)
        ax2.axvline(steps / days * (day + 1), color='k', linestyle='-', alpha=0.1)


    x_ticks_labels = ['00:00','06:00', '12:00','18:00','00:00','06:00', '12:00','18:00','00:00','06:00', '12:00','18:00','00:00','06:00', '12:00','18:00','00:00','06:00', '12:00','18:00','00:00',]
    ax1.set_xticks(np.arange(min(x_steps), max(x_steps), 36))
    ax1.set_xticklabels(x_ticks_labels, rotation='horizontal' , fontsize=8)
    ax2.set_xticks(np.arange(min(x_steps), max(x_steps), 36))
    ax2.set_xticklabels(x_ticks_labels, rotation='horizontal', fontsize=8)

    ax1.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=4, fontsize=8)
    ax2.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=3, fontsize=8)

    ax1.set_xlabel('time', fontsize=8)
    ax1.set_ylabel('kWh', fontsize=8)
    ax2.set_xlabel('time', fontsize=8)
    ax2.set_ylabel('kWh', fontsize=8)

    ax1.set_title('SOC of agents battery ', y=1.12, fontsize=10)
    ax2.set_title('Mean deficit', y=1.12, fontsize=10)

    ax1.set_ylim([0, max_battery*1.1])
    ax2.set_ylim([0, max_deficit*1.1])

    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    fig_soc_preferred.savefig('/Users/dirkvandenbiggelaar/Desktop/python_plots/fig_soc_preferred.png', bbox_inches='tight')


def plot_control_values(w_sharing_factors_list_over_time, E_actual_supplied_list_over_time, E_demand_list_over_time, c_prices_over_time, number_of_buyers_over_time, number_of_sellers_over_time, steps, N):

    fig_control_values = plt.figure(figsize=figsize_double, dpi=500)
    plt.tight_layout(pad=4, w_pad=10, h_pad=10)
    x_steps = np.arange(steps)


    mean_sharing_factors = np.mean(w_sharing_factors_list_over_time, axis=0)
    std_sharing_factors = np.std(w_sharing_factors_list_over_time, axis=0)
    mean_c_prices_over_time = np.mean(c_prices_over_time, axis=0)
    std_c_prices = np.std(c_prices_over_time, axis=0)
    E_actual_supplied_over_time = np.sum(E_actual_supplied_list_over_time, axis=0)
    E_demand_over_time = np.sum(E_demand_list_over_time, axis=0)
    max_supply = max(E_actual_supplied_over_time)
    max_demand = max(E_demand_over_time)
    max_price = max(mean_c_prices_over_time + max(std_c_prices))


    ax1 = fig_control_values.add_subplot(221)
    ax2 = fig_control_values.add_subplot(224)
    ax3 = fig_control_values.add_subplot(223)
    ax4 = fig_control_values.add_subplot(222)

    ax1.plot(E_actual_supplied_over_time, label="total supply")
    ax1.plot(E_demand_over_time, label="total demand")

    ax2.plot(mean_sharing_factors, label="mean sharing factors")
    ax2.fill_between(x_steps, mean_sharing_factors - std_sharing_factors, mean_sharing_factors + std_sharing_factors, color=c, alpha=0.3, label='SD of sharing-factors')

    ax3.plot(mean_c_prices_over_time, label="mean bidding-price")
    ax3.fill_between(x_steps, mean_c_prices_over_time - std_c_prices, mean_c_prices_over_time + std_c_prices, color=c, alpha=0.3, label='SD of bidding-prices')

    ax4.plot(number_of_buyers_over_time, label="buyers")
    ax4.plot(number_of_sellers_over_time, label="sellers")

    for day in range(days):
        ax1.axvline(steps/days * (day + 1), color='k', linestyle='-', alpha=0.1)
        ax2.axvline(steps/days * (day + 1), color='k', linestyle='-', alpha=0.1)
        ax3.axvline(steps/days * (day + 1), color='k', linestyle='-', alpha=0.1)
        ax4.axvline(steps/days * (day + 1), color='k', linestyle='-', alpha=0.1)

    x_ticks_labels = ['00:00', '12:00','00:00', '12:00','00:00', '12:00','00:00', '12:00','00:00', '12:00','00:00']
    ax1.set_xticks(np.arange(min(x_steps), max(x_steps), 72))
    ax1.set_xticklabels(x_ticks_labels, rotation='horizontal', fontsize=8)
    ax2.set_xticks(np.arange(min(x_steps), max(x_steps), 72))
    ax2.set_xticklabels(x_ticks_labels, rotation='horizontal', fontsize=8)
    ax3.set_xticks(np.arange(min(x_steps), max(x_steps), 72))
    ax3.set_xticklabels(x_ticks_labels, rotation='horizontal', fontsize=8)
    ax4.set_xticks(np.arange(min(x_steps), max(x_steps), 72))
    ax4.set_xticklabels(x_ticks_labels, rotation='horizontal', fontsize=8)

    ax1.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=3, fontsize=8)
    ax2.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=3, fontsize=8)
    ax3.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=3, fontsize=8)
    ax4.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=3, fontsize=8)

    ax1.set_xlabel('time', fontsize=8)
    ax1.set_ylabel('kWh', fontsize=8)
    ax2.set_xlabel('time', fontsize=8)
    ax2.set_ylabel('factor', fontsize=8)
    ax3.set_xlabel('time', fontsize=8)
    ax3.set_ylabel('price', fontsize=8)
    ax4.set_xlabel('time', fontsize=8)
    ax4.set_ylabel('pool size', fontsize=8)

    ax1.set_title('Supply and Demand', y=1.12, fontsize=10)
    ax2.set_title('Sharing-factor', y=1.12, fontsize=10)
    ax3.set_title('Bidding-price', y=1.12, fontsize=10)
    ax4.set_title('Sellers and Suyers', y=1.12, fontsize=10)

    ax1.set_ylim([0, max(max_supply, max_demand)*1.1])
    ax2.set_ylim([0, 1])
    ax3.set_ylim([0, max_price*1.1])
    ax4.set_ylim([0, N])

    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    fig_control_values.savefig('/Users/dirkvandenbiggelaar/Desktop/python_plots/fig_control_values.png', bbox_inches='tight')  # save the figure to file


def plot_supplied_vs_surplus_total(actual_batteries_over_time, surplus_on_step_over_time, E_total_supply_over_time, E_actual_supplied_list_over_time, E_demand_over_time, N, steps):

    fig_supplied_vs_surplus_total = plt.figure(figsize=figsize_single)
    plt.tight_layout(w_pad=0.2, h_pad=0.4)
    x_steps = np.arange(steps)

    E_actual_supplied_total_over_time = np.sum(E_actual_supplied_list_over_time, axis=0)
    max_value = max([max(surplus_on_step_over_time), max(E_actual_supplied_total_over_time), max(E_demand_over_time)])
    ax1 = fig_supplied_vs_surplus_total.add_subplot(111)

    ax1.plot(surplus_on_step_over_time, label='total surplus')
    ax1.plot(E_actual_supplied_total_over_time, label='total supplied')
    ax1.plot(E_demand_over_time, label='total demand')
    ax1.set_title('energy availability')

    for day in range(days):
        ax1.axvline(steps/days * (day + 1), color='k', linestyle='-', alpha=0.1)

    x_ticks_labels = ['00:00', '12:00', '00:00', '12:00', '00:00', '12:00', '00:00', '12:00', '00:00', '12:00', '00:00']
    ax1.set_xticks(np.arange(min(x_steps), max(x_steps), 72))
    ax1.set_xticklabels(x_ticks_labels, rotation='horizontal', fontsize=8)

    ax1.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=3, fontsize=8)

    ax1.set_xlabel('time', fontsize=8)
    ax1.set_ylabel('kWh', fontsize=8)

    ax1.set_title('Supply and Demand', y=1.12, fontsize=10)

    ax1.set_ylim([0, max_value * 1.1])

    fig_supplied_vs_surplus_total.savefig('/Users/dirkvandenbiggelaar/Desktop/python_plots/fig_supplied_vs_surplus_total.png', bbox_inches='tight')  # save the figure to file


def plot_iterations(global_iteration_over_time, buyer_iteration_over_time, seller_iteration_over_time, steps):

    fig_plot_iterations = plt.figure(figsize=figsize_single, dpi=500)
    plt.tight_layout(w_pad=0.2, h_pad=0.4)
    x_steps = np.arange(steps)

    ax1 = fig_plot_iterations.add_subplot(111)

    ax1.plot(global_iteration_over_time, label='global-level')
    ax1.plot(buyer_iteration_over_time, label='buyers-level')
    ax1.plot(seller_iteration_over_time, label='sellers-level')

    for day in range(days):
        ax1.axvline(steps/days * (day + 1), color='k', linestyle='-', alpha=0.1)

    x_ticks_labels = ['00:00', '12:00','00:00', '12:00','00:00', '12:00','00:00', '12:00','00:00', '12:00','00:00']
    ax1.set_xticks(np.arange(min(x_steps), max(x_steps), 72))
    ax1.set_xticklabels(x_ticks_labels, rotation='horizontal', fontsize=8)

    ax1.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=3, fontsize=8)

    ax1.set_title('Game iterations', y=1.06, fontsize=10)

    ax1.set_xlabel('time', fontsize=8)
    ax1.set_ylabel('kWh', fontsize=8)

    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    fig_plot_iterations.savefig('/Users/dirkvandenbiggelaar/Desktop/python_plots/fig_plot_iterations.png', bbox_inches='tight')  # save the figure to file


def plot_costs_over_time(E_demand_list_over_time, E_allocated_list_over_time, payment_list, E_total_supply_list_over_time, E_actual_supplied_list, revenue_list, profit_list_over_time, profit_list_summed_over_time, N, steps):
    """ Costs over time """

    """ Zooms in on 1 agent over time"""
    fig_costs_over_time_agent = plt.figure(figsize=(figsize_double), dpi=500)
    plt.tight_layout(w_pad=0.2, h_pad=0.4)
    x_steps = np.arange(steps)

    sales = fig_costs_over_time_agent.add_subplot(311)
    buys = fig_costs_over_time_agent.add_subplot(312)
    profit = fig_costs_over_time_agent.add_subplot(313)

    agent = np.arange(0,N)

    for i in range(len(agent)):
        buys.plot(payment_list[agent[i]][:], label='payments from agent' + str(agent[i]))
        sales.plot(revenue_list[agent[i]][:], label='revenue to agent' + str(agent[i]))

    balance_over_time = np.zeros(steps)
    for i in range(len(agent)):
        balance = 0
        for step in range(steps):
            balance += revenue_list[agent[i]][step] - payment_list[agent[i]][step]
            balance_over_time[step] = balance

    profit.plot(balance_over_time, label = 'balance per agent')

    for day in range(days):
        buys.axvline(steps / days * (day + 1), color='k', linestyle='-', alpha=0.1)
        sales.axvline(steps / days * (day + 1), color='k', linestyle='-', alpha=0.1)
        profit.axvline(steps / days * (day + 1), color='k', linestyle='-', alpha=0.1)


    x_ticks_labels = ['00:00','06:00', '12:00','18:00','00:00','06:00', '12:00','18:00','00:00','06:00', '12:00','18:00','00:00','06:00', '12:00','18:00','00:00','06:00', '12:00','18:00','00:00',]
    buys.set_xticks(np.arange(min(x_steps), max(x_steps), 36))
    buys.set_xticklabels(x_ticks_labels, rotation='horizontal', fontsize=8)
    sales.set_xticks(np.arange(min(x_steps), max(x_steps), 36))
    sales.set_xticklabels(x_ticks_labels, rotation='horizontal', fontsize=8)
    profit.set_xticks(np.arange(min(x_steps), max(x_steps), 36))
    profit.set_xticklabels(x_ticks_labels, rotation='horizontal', fontsize=8)

    buys.set_title('Payments per agent')
    sales.set_title('Sales per agent')
    profit.set_title('Profit')

    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    fig_costs_over_time_agent.savefig('/Users/dirkvandenbiggelaar/Desktop/python_plots/fig_costs_over_time_agent.png', bbox_inches='tight')  # save the figure to file









""" PSO AND OTHER EMBEDDED STUFF """
def plot_PSO(results_over_time, P_supply_list_over_time, P_demand_list_over_time, gen_output_list_over_time, load_demand_list_over_time, avg_battery_soc_list_over_time, N, sim_steps):

    fig_plot_PSO_dispatch = plt.figure(figsize=(20,5))
    ax1 = fig_plot_PSO_dispatch.add_subplot(211)
    ax2 = fig_plot_PSO_dispatch.add_subplot(212)

    total_supply_on_step = np.zeros(sim_steps)
    for step in range(sim_steps):
        supply_per_agent_on_step = 0
        for agent in range(N):
            supply_per_agent_on_step += results_over_time[agent][step]
        total_supply_on_step[step] = supply_per_agent_on_step
        # ax1.plot(results_over_time[agent][:])

    ax1.plot(total_supply_on_step, label='Generator dispatch optimized')
    ax1.plot(P_supply_list_over_time, label='Generator dispatch maximum')
    ax1.plot(P_demand_list_over_time, label='Demand of households total')

    ax2.plot(gen_output_list_over_time, label='gen_output_list_over_time')
    ax2.plot(load_demand_list_over_time, label='load_demand_list_over_time')
    ax2.plot(avg_battery_soc_list_over_time, label='mean battery over time')

    plt.legend()

    fig_plot_PSO_dispatch.savefig('/Users/dirkvandenbiggelaar/Desktop/python_plots/fig_plot_PSO_dispatch.png', bbox_inches='tight')  # save the figure to file
def plot_C_P(load_series_total, production_series_total):
    fig_total_P_C = plt.figure(figsize=(20,5))
    plt.plot(load_series_total, label='total production')
    plt.plot(production_series_total, label='total production')

    x_position = [1,2,3,4,5]
    for i in range(len(x_position)):
        plt.axvline(x_position[i]*144, color='k', linestyle='--', alpha=0.3)

    plt.legend()
    plt.suptitle('Total Production vs. Consumption')
    fig_total_P_C.savefig('/Users/dirkvandenbiggelaar/Desktop/python_plots/fig_total_P_C.png', bbox_inches='tight')  # save the figure to file

    return
def plot_input_data(big_data_file, sim_steps,N):

    fig_input_data = plt.figure(figsize=(10,10))
    ax1 = fig_input_data.add_subplot(311)
    ax2 = fig_input_data.add_subplot(312)
    ax3 = fig_input_data.add_subplot(313)

    load_series = np.zeros(sim_steps)
    production_series = np.zeros(sim_steps)

    load_series_total = np.zeros(sim_steps)
    production_series_total = np.zeros(sim_steps)

    for agent in range(N):
        max_consumption = max(big_data_file[:][agent][0])

        for i in range(sim_steps):
            load_series[i] = big_data_file[i][agent][0]
            production_series[i] = big_data_file[i][agent][1]
            load_series_total[i] += load_series[i]
            production_series_total[i] += production_series[i]

        ax1.plot(load_series, label='load of agent' + str(int(agent)))
        ax2.plot(production_series, label='production of agent' + str(int(agent)))

    ax3.plot(load_series_total, label='total load')
    ax3.plot(production_series_total, label='total production')



    ax1.set_title('consumption per agent')
    ax2.set_title('production per agent')
    ax3.set_title('total production and total consumption')
    ax3.legend()

    plt.suptitle('Supply vs Demand')
    fig_input_data.savefig('/Users/dirkvandenbiggelaar/Desktop/python_plots/fig_input_data.png', bbox_inches='tight')  # save the figure to file


    load = sum(load_series_total)
    production = sum(production_series_total)

    return load, production, load_series_total, production_series_total





""" TODO 

prediction analyseren
congestion set-up maken


"""
