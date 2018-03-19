import csv
import pdb
import random
import math

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.dates as md
import datetime as dt
fig_width = (10,3)

""" kleur TU-DELFT bies: 00A6D6"""
sns.set()
import numpy as np
x = 100
z = 10
w_range = np.arange(x)/x
factor_revenue_range = np.arange(z)/z
print(factor_revenue_range)

""" in a grid of 10 agents all supplying equal energy"""

fig_noprediction_utilityplot = plt.figure(figsize=(fig_width), dpi=250)
plt.tight_layout(w_pad=0.2, h_pad=0.4)

ax1 = fig_noprediction_utilityplot.add_subplot(111)

plot = 0
for j in range(len(factor_revenue_range)):
    plot += 1
    E_j_opt = 10
    R_d_opt = 50
    E_d_opt = 50

    factor_revenue = factor_revenue_range[j]

    utility_j_over_time = np.zeros(x)
    degradation = np.zeros(x)
    revenue = np.zeros(x)
    if plot > 8:
        break
    for i in range(len(w_range)):
        w = w_range[i]
        utility_j_over_time[i] =  np.log(1 + E_j_opt * (1 - w)) + factor_revenue * (R_d_opt * (E_j_opt * w / (E_d_opt + E_j_opt * w)))
        degradation[i]
        revenue[i]
    ax1.plot(w_range, utility_j_over_time, label='factor = ' + str(factor_revenue))

ax1.set_xlabel('sharing factor', fontsize=8)
ax1.set_ylabel('utility seller', fontsize=8)

ax1.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=4, fontsize=6)

plt.subplots_adjust(wspace=0.1, hspace=0.5)
plt.show()
fig_noprediction_utilityplot.savefig('/Users/dirkvandenbiggelaar/Desktop/used_plots/fig_noprediction_utilityplot.png', bbox_inches='tight')
