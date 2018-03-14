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
e=sns.color_palette()[4]


fig_width = (13,6)
figsize_single = (13,2.5)
figsize_double = (13,6)

days = 5



EBZ_deficit = np.load('/Users/dirkvandenbiggelaar/Desktop/result_files/deficits_N40/EBZ_N40_deficit_total_over_time_avg.npy')
no_prediction_deficit = np.load('/Users/dirkvandenbiggelaar/Desktop/result_files/deficits_N40/no_prediction_N40_deficit_total_over_time_avg.npy')
no_trading_deficit = np.load('/Users/dirkvandenbiggelaar/Desktop/result_files/deficits_N40/no_trading_N40_deficit_total_over_time_avg.npy')
supply_all_deficit = np.load('/Users/dirkvandenbiggelaar/Desktop/result_files/deficits_N40/supply_all_N40_deficit_total_over_time_avg.npy')



fig_compare_deficits = plt.figure(figsize=figsize_single, dpi=500)
plt.tight_layout(w_pad=0.2, h_pad=0.4)


plt.plot(EBZ_deficit, color=c, label='EBZ')
# plt.plot(no_prediction_deficit, color=a, label='no prediction')
plt.plot(no_trading_deficit, color=d, label='no trading')
plt.plot(supply_all_deficit, color=e, label='supplying all surplus')

plt.legend(loc='lower right', bbox_to_anchor=(1, -0.4), ncol=3)

fig_compare_deficits.savefig('/Users/dirkvandenbiggelaar/Desktop/used_plots/fig_compare_deficits.png', bbox_inches='tight')
