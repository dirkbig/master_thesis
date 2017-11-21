import csv
import numpy as np
from scipy.optimize import minimize
import random


def utility_seller(x, sign=-1):
    return np.log(1.0 + 10 * (1 - x))


sol_seller = minimize(utility_seller, initial_conditions_seller, method='SLSQP', bounds=bounds_seller,
                      constraints=cons_seller)  # bounds=bounds
