import matplotlib.pyplot as plt
import numpy as np


Ej = 200
alpha = 1



Rd = 500.80204398
E_others_d =  200.5

Rp = 300.534
E_others_p = 200.0


w = np.arange(0, 1, 0.01) # Grid of 0.01 spacing from -2 to 10


""" Share of total revenue in direct sales for j with selling factor w"""
direct_revenue_for_agent = Rd * (Ej*w)/(E_others_d + (Ej*w))

""" Share of total revenue in predicted sales for j with selling factor w"""
predicted_revenue_for_agent = Rp * (Ej*(1-w))/(E_others_p + (Ej*(1-w)))




""" missed revenue on 
selling """
y_direct = direct_revenue_for_agent**0.8

""" missed revenue on 
storing """
y_prediction = predicted_revenue_for_agent**0.8

plt.title("Utility of Sellers")
plt.xlabel('w of j')
plt.ylabel('U of j')
plt.plot(y_direct, label="direct U" )
plt.plot(y_prediction, label="predicted U")
plt.plot(y_prediction + y_direct, label="total U")
plt.legend()
plt.show()


direct_revenue_for_agent = Rd * (Ej*w)/(E_others_d + (Ej*w))
predicted_revenue_for_agent = Rp * (Ej*(1-w))/(E_others_p + (Ej*(1-w)))

# plt.plot(y_prediction + y_direct)

