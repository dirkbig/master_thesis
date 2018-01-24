import matplotlib.pyplot as plt
import numpy as np

def plot_E_total_surplus_prediction_per_step(E_total_surplus_prediction_per_step, N):
    # plt.plot(E_total_surplus_prediction_per_step/N)
    # plt.show()
    return


def plot_results(mean_sharing_factors, supply_over_time_list, demand_over_time, c_nominal_over_time, buyers, sellers):
    plt.subplot(4, 1, 1)
    plt.plot(supply_over_time_list, label="supply_over_time_list")
    plt.plot(demand_over_time, label="demand_over_time")
    plt.title('supply/demand over time')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(mean_sharing_factors, label="mean_sharing_factors")
    plt.title('mean sharing factor per step')
    plt.legend()


    plt.subplot(4, 2, 1)
    plt.plot(c_nominal_over_time, label="c_nominal_over_time")
    plt.title('bidding price over time')
    plt.legend()

    plt.subplot(4, 2, 2)
    plt.plot(buyers, label="buyers")
    plt.plot(sellers, label="sellers")
    plt.title('sellers/buyers pool')
    plt.legend()

    plt.suptitle('Results')
    plt.show()

def plot_prediction(means_surplus, means_load):
    # plt.title('mean of surplus and load')
    # plt.plot(means_surplus)
    # plt.plot(means_load)
    # plt.show
    return

def plot_w_nominal_progression(w_nominal_over_time, R_prediction_over_time, E_prediction_over_time, E_real_over_time, R_real_over_time, c_nominal):
    """w_nominal against predicted energy and predicted revenue"""

    R_prediction_over_time_normalised = R_prediction_over_time/max(R_prediction_over_time)
    E_prediction_over_time_normalised = E_prediction_over_time/max(E_prediction_over_time)
    E_real_over_time_normalised = E_real_over_time/max(E_real_over_time)
    R_real_over_time_normalised = R_real_over_time/max(R_real_over_time)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(211)
    ax1.plot(w_nominal_over_time, label='w_nominal')
    ax1.plot(E_prediction_over_time, label='E_prediction (weighted)')
    ax1.plot(c_nominal, label='c_nominal')
    ax1.plot(E_real_over_time, label='E_real')
    ax1.set_title('Energy surplus real vs prediction model')
    ax1.legend()


    ax2 = fig1.add_subplot(212)
    ax2.plot(w_nominal_over_time, label='w_nominal')
    ax2.plot(R_prediction_over_time, label='R_prediction (weighted)')
    ax2.plot(R_real_over_time, label='R_real')
    ax1.set_title('Revenue real vs prediction model')
    ax2.legend()

    plt.suptitle('Normalised data on trading')
    #plt.subplots_adjust(left=0.2, wspace=0.8, top=0.8)
    plt.show()

    """normalised"""
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(211)
    # ax1.plot(w_nominal_over_time, label='w_nominal')
    # ax1.plot(E_prediction_over_time_normalised, label='E_prediction')
    # ax1.plot(E_real_over_time_normalised, label='E_real')
    # ax1.set_title('Energy surplus real vs prediction model')
    # ax1.legend()
    #
    #
    # ax2 = fig1.add_subplot(212)
    # ax2.plot(w_nominal_over_time, label='w_nominal')
    # ax2.plot(R_prediction_over_time_normalised, label='R_prediction')
    # ax2.plot(R_real_over_time_normalised + 1, label='R_real')
    # ax1.set_title('Revenue real vs prediction model')
    # ax2.legend()
    #
    # plt.suptitle('Normalised data on trading')
    # #plt.subplots_adjust(left=0.2, wspace=0.8, top=0.8)
    # plt.show()


    # fig.savefig(file_name, bbox_inches='tight')

def plot_available_vs_supplied(actual_batteries_over_time, E_total_supply_over_time, E_demand_over_time, N):

    fig_available_vs_supplied = plt.figure()
    ax1 = fig_available_vs_supplied.add_subplot(211)
    ax1.plot(E_demand_over_time, label='Demanded by buyers')
    ax1.plot(E_total_supply_over_time, label='Supplied by sellers')
    ax1.set_title('Demanded vs Supplied')
    ax1.legend()

    batt_soc_ax2 = fig_available_vs_supplied.add_subplot(212)
    for agent in range(N):
        batt_soc_ax2.plot(actual_batteries_over_time[agent], label='soc agent %d' % agent)
    batt_soc_ax2.set_title('State-of-Charge batteries per agent')
    batt_soc_ax2.legend()

    plt.suptitle('Battery business')
    plt.show()

def plot_supplied_vs_surplus_total(surplus_on_step_over_time, supplied_on_step_over_time, demand_on_step_over_time):

    fig_supplied_vs_surplus_total = plt.figure()

    ax1 = fig_supplied_vs_surplus_total.add_subplot(211)
    ax1.plot(surplus_on_step_over_time, label='surplus energy in the system')
    ax1.plot(supplied_on_step_over_time, label='supplied in the system')
    ax1.plot(demand_on_step_over_time, label='demand in the system')
    ax1.legend()
    ax1.set_title('energy availability')

    plt.show()




def plot_utilities(utilities_buyers_over_time, utilities_sellers_over_time, N, steps): #utility_buyers_over_time[N][3][sim_steps]


    utilities_buyers_over_time_series = np.zeros(len(utilities_buyers_over_time))

    # for i in range(len(utilities_buyers_over_time)):
    #     utilities_buyers_over_time_series[i] = utilities_buyers_over_time

    # mean_direct_utility_over_agents = np.zeros(len())
    # mean_predicted_utility_over_agents
    #
    # for i in len(utilities_sellers_over_time):
    #     for agent in range(N):
    #         mean_direct_utility_over_agents[i] = sum(utilities_sellers_over_time[agent][3][i])/N
    #         mean_predicted_utility_over_agents[i] = sum(utilities_sellers_over_time[agent][2][i])/N

    utilities_sellers_over_time_1 = np.zeros(steps)
    utilities_sellers_over_time_2 = np.zeros(steps)
    utilities_sellers_over_time_3 = np.zeros(steps)

    utilities_buyers_total_series = np.zeros(steps)
    utilities_buyers_demand_gap_series = np.zeros(steps)
    utilities_buyers_costs_series = np.zeros(steps)


    # print(np.shape(utilities_sellers_over_time))
    fig_utilities = plt.figure()

    ax1 = fig_utilities.add_subplot(211)
    for agent in range(N-3):

        for i in range(steps):
            utilities_sellers_over_time_1[i] = utilities_sellers_over_time[i][agent][0]
            utilities_sellers_over_time_2[i] = utilities_sellers_over_time[i][agent][1]
            utilities_sellers_over_time_3[i] = utilities_sellers_over_time[i][agent][2]

        # ax1.plot(utilities_sellers_over_time_1, label='Utility sellers total' + str(int(agent)))
        # ax1.plot(utilities_sellers_over_time_2, label='Utility sellers prediction part' + str(int(agent)))
        # ax1.plot(utilities_sellers_over_time_3, label='Utility sellers direct part' + str(int(agent)))

        ax1.plot(utilities_sellers_over_time_1 - utilities_sellers_over_time_3, label='Utility sellers total' + str(int(agent)))

        # ax1.plot(utilities_sellers_over_time[:][agent][1], label='Utility sellers total' + str(int(agent)))
        # ax1.plot(utilities_sellers_over_time[:][agent][2], label='Utility sellers prediction part' + str(int(agent)))
        # ax1.plot(utilities_sellers_over_time[:][agent][3], label='Utility sellers direct part' + str(int(agent)))
    ax1.set_title('plotted utility sellers')
    ax1.legend()

    ax2 = fig_utilities.add_subplot(212)
    for agent in range(N-3):

        for i in range(steps):
            utilities_buyers_total_series[i] = utilities_buyers_over_time[i][agent][0]
            utilities_buyers_demand_gap_series[i] = utilities_buyers_over_time[i][agent][2]
            utilities_buyers_costs_series[i] = utilities_buyers_over_time[i][agent][3]

        ax2.plot(utilities_buyers_total_series, label='Utility buyers total')
        ax2.plot(utilities_buyers_demand_gap_series, label='Utility buyers demand gap')
        ax2.plot(utilities_buyers_costs_series, label='Utility buyers costs')
    ax2.set_title('plotted utility buyers')
    ax2.legend()

    plt.suptitle('Utility values Buyers/Sellers')
    plt.show()

    # ax2 = fig_utilities.add.subplot(212)

def plot_avg_soc_preferred(soc_preferred_list_over_time, avg_soc_preferred_over_time):

    fig_soc_preferred = plt.figure()
    ax1 = fig_soc_preferred.add_subplot(211)


    ax1.plot(avg_soc_preferred_over_time, label='average soc over time')
    ax1.legend()

    plt.show()

def plot_supply_demand(surplus_in_grid_over_time, demand_in_grid_over_time, N):

    fig_supply_demand = plt.figure()
    ax1 = fig_supply_demand.add_subplot(211)
    ax2 = fig_supply_demand.add_subplot(212)

    total_surplus = sum(surplus_in_grid_over_time)
    total_demand = sum(demand_in_grid_over_time)

    for agent in range(N):
        ax1.plot(surplus_in_grid_over_time, label='surplus of agent ' + str(int(agent)))
        ax2.plot(demand_in_grid_over_time, label='demand of agent ' + str(int(agent)))

    ax1.plot(total_surplus, label='total surplus')
    ax2.plot(total_demand, label='total demand')
    ax1.legend()
    ax2.legend()

    plt.suptitle('Supply vs Demand')
    plt.show


def plot_input_data(big_data_file, sim_steps,N):

    fig_input_data = plt.figure()
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

        for i in range(sim_steps):
            load_series_total[i] += load_series[i]
            production_series_total[i] += production_series[i]

        ax1.plot(load_series, label='load of agent' + str(int(agent)))
        ax2.plot(production_series, label='production of agent' + str(int(agent)))

    ax3.plot(load_series_total, label='total production')
    ax3.plot(production_series_total, label='total production')

    ax1.legend()
    ax2.legend()

    plt.suptitle('Supply vs Demand')
    # plt.show()


    load = sum(load_series_total)
    production = sum(production_series_total)

    return load, production, load_series_total, production_series_total


def plot_C_P(load_series_total, production_series_total):
    fig_total_P_C = plt.figure(figsize=(20,5))
    plt.plot(load_series_total, label='total production')
    plt.plot(production_series_total, label='total production')

    x_position = [1,2,3,4,5]
    for i in range(len(x_position)):
        plt.axvline(x_position[i]*144, color='k', linestyle='--', alpha=0.3)

    plt.legend()
    plt.suptitle('Total Production vs. Consumption')
    fig_total_P_C.savefig('/Users/dirkvandenbiggelaar/Desktop/python_plots/fig_total_P_C.pdf', bbox_inches='tight')  # save the figure to file

    return
""" TODO 

prediction analyseren
congestion set-up maken


"""
