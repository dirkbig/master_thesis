from matplotlib.ticker import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()


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

    # def millions(x, pos):
    #     'The two args are the value and tick position'
    #     return '$%1.1fM' % (x * 1e-6)
    #
    # formatter = FuncFormatter(millions)

    fig_thesis_iteration = plt.figure(figsize=(20, 5))
    iterations = fig_thesis_iteration.add_subplot(111)


    # ax.yaxis.set_major_formatter(formatter)
    iterations.bar(x, iter_global,width= 0.3, align='edge')
    iterations.bar(x, iter_buyer,width= -0.3, align='edge')
    iterations.bar(x, iter_seller,width= 0.3, align='center')

    # iterations.xticks(x, ('Bill', 'Fred', 'Mary', 'Sue'))
    # plt.show()


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
            # supply.plot(w_nominal_over_time_batch[batch]*30, 'k-')
    print('oh hey')

    plt.show()