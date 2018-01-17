import numpy as np


horizon = 3
E_prediction_series = [1,2,3,4,5,6]

future_availability = sum(range(horizon), E_prediction_series)

print(future_availability)