from matplotlib.ticker import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colors as mcolors
sns.set()
""" kleur TU-DELFT bies: 00A6D6"""
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
c=sns.color_palette()[0]

def thesis_iteration_plot(list_mean_iterations_batch, num_batches, num_agents, agents_low, agents_high):
    """ increase in complexity when increasing agents """
    iter_global = np.zeros(num_batches)
    iter_buyer = np.zeros(num_batches)
    iter_seller = np.zeros(num_batches)
    x = np.arange(agents_low, agents_high)
    for i in range(num_batches):
        iter_global[i] = list_mean_iterations_batch[i][0]
        iter_buyer[i] = list_mean_iterations_batch[i][1]
        iter_seller[i] = list_mean_iterations_batch[i][2]

    fig_thesis_iteration = plt.figure(figsize=(20, 5))
    iterations = fig_thesis_iteration.add_subplot(111)


    iterations.bar(x, iter_global,width= 0.3, align='edge')
    iterations.bar(x, iter_buyer,width= -0.3, align='edge')
    iterations.bar(x, iter_seller,width= 0.3, align='center')



def thesis_control_values_plot(c_nominal_over_time_batch, w_nominal_over_time_batch, num_batches):

    fig_thesis_control_values = plt.figure(figsize=(20, 5))
    c = fig_thesis_control_values.add_subplot(211)
    w = fig_thesis_control_values.add_subplot(212)
    for batch in range(num_batches):
        if batch % 3 == 0:
            c.plot(c_nominal_over_time_batch[batch],'k-' , alpha=batch/(1.4*num_batches))
            w.plot(w_nominal_over_time_batch[batch],'k-', alpha=batch/(1.4*num_batches))


    plt.show()

    x = np.linspace(0, 100)
    y = np.sin(x / 10)
    std = np.random.rand(x.shape[0])

    plt.figure()
    plt.fill_between(x, y - std, y + std, alpha=0.3)
    plt.plot(x, y)
    plt.show()

def thesis_supply_demand_batch_plot(E_total_supply_list_over_time_mean, w_nominal_over_time_batch, num_batches):

    fig_thesis_supply_demand_batch = plt.figure(figsize=(20, 5))
    supply = fig_thesis_supply_demand_batch.add_subplot(111)

    for batch in range(num_batches):
        if batch % 3 == 0:
            supply.plot(E_total_supply_list_over_time_mean[batch], 'c-')

    supply.plot(w_nominal_over_time_batch[batch], 'k-')

    print('oh hey')

    plt.show()

def thesis_soc_batch_plot(actual_batteries_list_over_time_batch, socs_preferred_over_time_batch, E_actual_supplied_total_batch, num_batches, agents_low, num_steps):
    actual_batteries_over_time_mean = np.zeros((num_batches, num_steps))
    socs_preferred_over_time = np.zeros((num_batches, num_steps))
    E_actual_supplied_total = np.zeros((num_batches, num_steps))

    min_actual_batteries_list_over_time = np.zeros((num_batches,num_steps))
    max_actual_batteries_list_over_time = np.zeros((num_batches,num_steps))
    std_actual_batteries_list_over_time = np.zeros((num_batches,num_steps))

    fig_thesis_soc_batch_plot = plt.figure(figsize=(20, 5))
    soc = fig_thesis_soc_batch_plot.add_subplot(111)
    print(np.shape(actual_batteries_list_over_time_batch))

    N = agents_low
    batch_row = 0
    for batch in range(num_batches):
        actual_batteries_list_over_time = actual_batteries_list_over_time_batch[batch_row:(batch_row + N)]
        socs_preferred_list_over_time = socs_preferred_over_time_batch[batch_row:(batch_row + N)]
        E_actual_supplied_total = E_actual_supplied_total_batch[batch_row:(batch_row + N)]
        for step in range(num_steps):
            actual_batteries_over_time_mean[batch][step] = 0
            max_actual_batteries_list_over_time[batch] = np.amax(actual_batteries_list_over_time, axis=0)
            min_actual_batteries_list_over_time[batch] = np.amin(actual_batteries_list_over_time, axis=0)
            std_actual_batteries_list_over_time[batch] = np.std(actual_batteries_list_over_time, axis=0)
            print(np.shape(actual_batteries_over_time_mean))
            for agent in range(N):
                actual_batteries_over_time_mean[batch][step] += np.mean(actual_batteries_list_over_time[agent][step])/N
                # max_actual_batteries_list_over_time[batch][step] = min(actual_batteries_list_over_time)

        """ plot over batches """
        if batch % 4 == 0:
            soc.plot(actual_batteries_over_time_mean[batch],color=c,alpha= batch/(num_batches))
            # soc.fill_between(range(num_steps), min_actual_batteries_list_over_time[batch], max_actual_batteries_list_over_time[batch],
            # color=c, alpha=0.1)
            soc.fill_between(range(num_steps), actual_batteries_over_time_mean[batch] - std_actual_batteries_list_over_time[batch], actual_batteries_over_time_mean[batch] + std_actual_batteries_list_over_time[batch],
                             color=c, alpha=0.1)

        batch_row += N
        N += + 1


    plt.show()

    x = np.linspace(0, 100)
    y = np.sin(x / 10)
    std = np.random.rand(x.shape[0])

    plt.figure()
    plt.fill_between(x, y - std, y + std, alpha=0.3)
    plt.plot(x, y)
    plt.show()