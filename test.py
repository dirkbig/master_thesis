import numpy as np

prediction_range = 10

horizon_factor_up = np.arange(0.5, 1.5, 10)
horizon_factor_down = np.arange(1.5, 0.5, prediction_range)

affine_functions = [horizon_factor_up, horizon_factor_down]

print(np.shape(horizon_factor_up))
factor = min(affine_functions)

print(factor)