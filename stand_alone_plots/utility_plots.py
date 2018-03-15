
import seaborn as sns
sns.set()
import numpy as np
from functions.function_file import *

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

""" kleur TU-DELFT bies: 00A6D6"""
sns.set()


fig_utilities = plt.figure(figsize=(10,2))
fig_buyer_malicious_info = plt.figure(figsize=(10,2))



lambda11 = 2
lambda12 = 1
lambda21 = 1.5
lambda22 = 1.3

lambda_set = [lambda11, lambda12, lambda21, lambda22]


"""PLOT: utility buyers"""

E_i_demand = 20
E_total_supply = 20
E_total_supply_malicious = 8

c_i_bidding_price = 10
c_bidding_prices_others = 30
c_i_range = np.linspace(0,20,100)

utility_costs_plot = np.zeros(len(c_i_range))
utility_demand_gap_plot = np.zeros(len(c_i_range))
utility_i_plot = np.zeros(len(c_i_range))
utility_costs_plot_m = np.zeros(len(c_i_range))
utility_demand_gap_plot_m = np.zeros(len(c_i_range))
utility_i_plot_m = np.zeros(len(c_i_range))

ax_i = fig_utilities.add_subplot(121)
ax3 = fig_buyer_malicious_info.add_subplot(121)
ax4_m = fig_buyer_malicious_info.add_subplot(122)

for i in range(len(c_i_range)):
    c_i_bidding_price = c_i_range[i]
    utility_i, demand_gap, utility_demand_gap, utility_costs = calc_utility_function_i(E_i_demand, E_total_supply, c_i_bidding_price, c_bidding_prices_others, lambda_set)
    utility_i_m, demand_gap_m, utility_demand_gap_m, utility_costs_m = calc_utility_function_i(E_i_demand, E_total_supply_malicious, c_i_bidding_price, c_bidding_prices_others, lambda_set)

    utility_costs_plot[i]  = utility_costs
    utility_demand_gap_plot[i] = utility_demand_gap
    utility_i_plot[i] = utility_i

    utility_costs_plot_m[i]  = utility_costs_m
    utility_demand_gap_plot_m[i] = utility_demand_gap_m
    utility_i_plot_m[i] = utility_i_m

min_i = min(utility_i_plot)
min_i_m = min(utility_i_plot_m)

for i in range(len(utility_i_plot)):
    if utility_i_plot[i] == min_i:
        position_min_i = c_i_range[i]

for i in range(len(utility_i_plot_m)):
    if utility_i_plot_m[i] == min_i_m:
        position_min_i_m = c_i_range[i]


ax_i.plot(c_i_range,utility_costs_plot, label='utility costs')
ax_i.plot(c_i_range,utility_demand_gap_plot, label='utility demand gap')
ax_i.plot(c_i_range,utility_i_plot, label='total utility')
ax_i.axhline(min_i, color='k', linestyle='--', alpha=0.3)
ax_i.axvline(position_min_i, color='k', linestyle='--', alpha=0.3)


ax3.plot(c_i_range,utility_costs_plot, label='costs')
ax3.plot(c_i_range,utility_demand_gap_plot, label='demand-gap')
ax3.plot(c_i_range,utility_i_plot, label='total')
ax3.axhline(min_i, color='k', linestyle='--', alpha=0.3)
ax3.axvline(position_min_i, color='k', linestyle='--', alpha=0.3)


ax4_m.plot(c_i_range,utility_costs_plot_m, label='costs')
ax4_m.plot(c_i_range,utility_demand_gap_plot_m, label='demand-gap')
ax4_m.plot(c_i_range,utility_i_plot_m, label='total')
ax4_m.axhline(min_i_m, color='k', linestyle='--', alpha=0.3)
ax4_m.axvline(position_min_i_m, color='k', linestyle='--', alpha=0.3)


# ax3.set_ylim([0, 15])
ax3.set_xlim([0, 20])
ax3.set_ylim([0, 400])
ax4_m.set_xlim([0, 20])
ax4_m.set_ylim([0, 400])



ax3.set_xlabel('bidding price', fontsize=10)
ax3.set_ylabel('Utility i', fontsize=10)
ax4_m.set_xlabel('bidding price', fontsize=10)
ax4_m.set_ylabel('Utility i', fontsize=10)


"""PLOT: utility sellers"""

id_seller = 1
E_j_seller = 20
R_direct = 1
E_supply_others = 100
R_prediction = 3.5
E_supply_prediction = 150
E_j_prediction_seller = 10
w_j_range = np.linspace(0,1,100)

utility_direct_plot = np.zeros(len(w_j_range))
utility_predicted_plot = np.zeros(len(w_j_range))
utility_j_plot = np.zeros(len(w_j_range))

lambda21 = 2
lambda22 = 2

lambda_set = [lambda11, lambda12, lambda21, lambda22]

ax_j = fig_utilities.add_subplot(122)
for i in range(len(c_i_range)):
    w_j_storage_factor = w_j_range[i]
    prediction_utility, direct_utility, utility_j = calc_utility_function_j(id_seller, E_j_seller, R_direct, E_supply_others, R_prediction, E_supply_prediction, w_j_storage_factor, E_j_prediction_seller,lambda_set)

    utility_direct_plot[i] = direct_utility
    utility_predicted_plot[i] = prediction_utility
    utility_j_plot[i] = utility_j

min_j = max(utility_j_plot)
for i in range(len(utility_j_plot)):
    if utility_j_plot[i] == min_j:
        position_min_j = w_j_range[i]

ax_j.plot(w_j_range,utility_direct_plot, label='direct')
ax_j.plot(w_j_range,utility_predicted_plot, label='predicted')
ax_j.plot(w_j_range,utility_j_plot, label='total')
ax_j.axhline(min_j, color='k', linestyle='--', alpha=0.3)
ax_j.axvline(position_min_j, color='k', linestyle='--', alpha=0.3)

ax_i.set_xlabel('bdding price', fontsize=10)
ax_i.set_ylabel('Utility i', fontsize=10)
ax_j.set_xlabel('sharing factor', fontsize=10)
ax_j.set_ylabel('Utility j', fontsize=10)




""" Add vertical line"""
ax_j.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=3, fontsize=9)
ax_i.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=3, fontsize=9)
ax3.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=3, fontsize=9)
ax4_m.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=3, fontsize=9)


fig_utilities.savefig('/Users/dirkvandenbiggelaar/Desktop/used_plots/fig_utilities_functions.png',bbox_inches='tight')
fig_buyer_malicious_info.savefig('/Users/dirkvandenbiggelaar/Desktop/used_plots/fig_buyer_malicious_info.png',bbox_inches='tight')





