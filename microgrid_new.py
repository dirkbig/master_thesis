from source.BatteryModel import *
from source.PvModel import *
import pandas as pd
from source.function_file import *
from source.initialization import *
from mesa import Agent, Model


"""All starting parameters are initialised"""
duration = 1  # 1440                # Duration of the sim (one full day or 1440 time_steps of 1 minute) !!10 if test!!
N = 5                               # N agents only
step_list = np.zeros([duration])

c_S = 4                                              # c_S is selling price of the microgrid
c_B = 2                                              # c_B is buying price of the microgrid
c_macro = (c_B, c_S)                                 # Domain of available prices for bidding player i
possible_c_i = range(c_macro[0], c_macro[1])         # domain of solution for bidding price c_i
