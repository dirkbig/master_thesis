import matplotlib.pyplot as plt
import numpy as np
from source.function_file import *






fig_utilities = plt.figure(figsize=(10,2))

"""PLOT: utility buyers"""

E_i_demand = 2
E_total_supply = 10
c_i_bidding_price = 8
c_bidding_prices_others = 20
c_i_range = np.linspace(0,5,100)

utility_costs_plot = np.zeros(len(c_i_range))
utility_demand_gap_plot = np.zeros(len(c_i_range))
utility_i_plot = np.zeros(len(c_i_range))

"""PLOT: utility sellers"""

id_seller = 1
E_j_seller = 20
R_direct = 1
E_supply_others = 100
R_prediction = 1.2
E_supply_prediction = 150
E_j_prediction_seller = 10
w_j_range = np.linspace(0,1,100)


utility_direct_plot = np.zeros(len(w_j_range))
utility_predicted_plot = np.zeros(len(w_j_range))
utility_j_plot = np.zeros(len(w_j_range))


ax_i = fig_utilities.add_subplot(122)
for i in range(len(c_i_range)):
    c_i_bidding_price = c_i_range[i]
    utility_i, demand_gap, utility_demand_gap, utility_costs = calc_utility_function_i(E_i_demand, E_total_supply, c_i_bidding_price, c_bidding_prices_others)

    utility_costs_plot[i]  = utility_costs
    utility_demand_gap_plot[i] = utility_demand_gap
    utility_i_plot[i] = utility_i

min_i = min(utility_i_plot)
for i in range(len(utility_i_plot)):
    if utility_i_plot[i] == min_i:
        position_min_i = c_i_range[i]


ax_i.plot(c_i_range,utility_costs_plot, label='utility_costs_plot')
ax_i.plot(c_i_range,utility_demand_gap_plot, label='utility_demand_gap_plot')
ax_i.plot(c_i_range,utility_i_plot, label='utility_i_plot')
ax_i.axhline(min_i, color='k', linestyle='--', alpha=0.3)
ax_i.axvline(position_min_i, color='k', linestyle='--', alpha=0.3)

plt.xlabel('c_i', fontsize=10)
plt.ylabel('Utility i', fontsize=10)
""" Add vertical line"""

plt.legend()


ax_j = fig_utilities.add_subplot(121)
for i in range(len(c_i_range)):
    w_j_storage_factor = w_j_range[i]
    prediction_utility, direct_utility, utility_j = calc_utility_function_j(id_seller, E_j_seller, R_direct, E_supply_others, R_prediction, E_supply_prediction, w_j_storage_factor, E_j_prediction_seller)

    utility_direct_plot[i] = direct_utility
    utility_predicted_plot[i] = prediction_utility
    utility_j_plot[i] = utility_j

min_j = max(utility_j_plot)
for i in range(len(utility_j_plot)):
    if utility_j_plot[i] == min_j:
        position_min_j = w_j_range[i]

ax_j.plot(w_j_range,utility_direct_plot, label='utility_direct_plot')
ax_j.plot(w_j_range,utility_predicted_plot, label='utility_predicted_plot')
ax_j.plot(w_j_range,utility_j_plot, label='utility_j_plot')
ax_j.axhline(min_j, color='k', linestyle='--', alpha=0.3)
ax_j.axvline(position_min_j, color='k', linestyle='--', alpha=0.3)

plt.xlabel('w_j', fontsize=10)
plt.ylabel('Utility j', fontsize=10)
""" Add vertical line"""

plt.legend()

fig_utilities.savefig('/Users/dirkvandenbiggelaar/Desktop/python_plots/fig_utilities.pdf',bbox_inches='tight')
plt.show()


# save the figure to file



