import matplotlib.pyplot as plt
import numpy as np

def plot_E_total_surplus_prediction_per_step(E_total_surplus_prediction_per_step, N):
    plt.plot(E_total_surplus_prediction_per_step/N)
    plt.show()


def plot_results(mean_sharing_factors, supply_over_time_list, demand_over_time, c_nominal_over_time, buyers, sellers):
    plt.subplot(4, 1, 1)
    plt.plot(supply_over_time_list, label="supply_over_time_list")
    plt.plot(demand_over_time, label="demand_over_time")
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(mean_sharing_factors, label="mean_sharing_factors")
    plt.legend()


    plt.subplot(4, 2, 1)
    plt.plot(c_nominal_over_time, label="c_nominal_over_time")
    plt.legend()

    plt.subplot(4, 2, 2)
    plt.plot(buyers, label="buyers")
    plt.plot(sellers, label="sellers")
    plt.legend()
    plt.show()



def plot_w_nominal_progression(w_nominal_over_time, R_prediction_over_time, E_prediction_over_time,E_real_over_time_normalised,R_real_over_time_normalised):
    """w_nominal against predicted energy and predicted revenue"""

    R_prediction_over_time_normalised = R_prediction_over_time/max(R_prediction_over_time)
    E_prediction_over_time_normalised = E_prediction_over_time/max(E_prediction_over_time)
    E_real_over_time_normalised = E_real_over_time_normalised/max(E_real_over_time_normalised)
    R_real_over_time_normalised = R_real_over_time_normalised/max(R_real_over_time_normalised)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(211)
    ax1.plot(w_nominal_over_time, label='w_nominal')
    ax1.plot(E_prediction_over_time_normalised, label='E_prediction')
    ax1.plot(E_real_over_time_normalised, label='E_prediction')
    ax1.legend()


    ax2 = fig1.add_subplot(212)
    ax2.plot(w_nominal_over_time, label='w_nominal')
    ax2.plot(R_prediction_over_time_normalised, label='R_prediction')
    ax2.plot(R_real_over_time_normalised, label='R_prediction')
    ax2.legend()



    plt.show()

    # fig.savefig(file_name, bbox_inches='tight')
