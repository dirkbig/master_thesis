import numpy as np
# from source.batchrunner import Batch_number_of_agents

"""All starting parameters are initialised"""
starting_point = 0
stopping_point = 7200 - starting_point - 4000
step_day = 1440
timestep = 5
days = 5
blockchain = 'off'

step_time = 5
total_steps = step_day*days
sim_steps = int(total_steps/step_time)

N = 20
comm_radius = 10

step_list = np.zeros([sim_steps])

c_S = 10                                             # c_S is selling price of the microgrid
c_B = 1                                              # c_B is buying price of the microgrid
c_macro = (c_B, c_S)                                 # Domain of available prices for bidding player i
possible_c_i = range(c_macro[0], c_macro[1])         # domain of solution for bidding price c_i

e_buyers = 0.001
e_sellers = 0.001
e_global = 0.01
e_cn = 0.01
e_supply = 0.1