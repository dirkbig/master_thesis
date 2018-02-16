""" Particle swarm optimization """
import numpy as np
from pyswarm import pso
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from source.function_file import *
from mesa import Agent, Model



""" Run PSO optimization"""

class HouseholdAgent_PSO(Agent):
    """All microgrid household(agents) should be generated here; initialisation of prosumer tools """

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.id = unique_id

        """ self.gen_output is household pv panel operating at max"""
        self.gen_output = 0
        self.load_demand = 0

        """ Household constraints"""
        self.ramp_max = 10 # Kw/timestep

        self.gen_output_min = 0
        self.gen_output_max = 0

        self.battery_soc = 300
        self.battery_ramp = 10
        self.battery_max = self.battery_soc
        self.battery_min = 0
        self.gen_battery = 0

        self.constraint_min = 0
        self.constraint_max = 0

        self.alpha = 10 #* np.random.random_sample()
        self.beta = 10 * np.random.random_sample()
        self.gamma = 1 * np.random.random_sample()

        self.costfunction = 0
        self.power_surplus_i = 0

        self.P_prev = 0

    def __repr__(self):
        return "ID: %d, pvgeneration:%d, consumption:%d" % (self.id, self.gen_output, self.load_demand)


class MicroGrid_PSO(Model):
    """create environment in which agents can operate"""

    def __init__(self, big_data_file, starting_point, N):
        """ Time is initialised"""
        self.steps = 0
        self.time = 0

        print("PSO validation Model")
        self.num_households = N
        self.agents = []

        """ agent creation """
        for i in range(self.num_households):
            agent = HouseholdAgent_PSO(i, self)
            self.agents.append(agent)

        self.data = DataAgent_PSO(self)

        self.costfunction_list = np.zeros(N)
        self.P_supply_list = np.zeros(N)
        self.P_demand_list = np.zeros(N)

        """ PSO """
        self.inertia_max = 0.9
        self.inertia_min = 0.4
        self.iter_max = 0
        self.iteration = 0
        self.inertia_weight = 0

        self.F_max = 0
        self.F_min = 0

        self.alpha_vector = np.zeros(N)
        self.beta_vector = np.zeros(N)
        self.gamma_vector = np.zeros(N)
        self.gen_output_list = np.zeros(N)
        self.load_demand_list = np.zeros(N)
        self.P_optimized_list = np.zeros(N)
        self.battery_soc_list = np.zeros(N)

    def pso_step(self, big_data_file, N):
        print('step ',self.steps)

        """ update values from data """
        for agent in self.agents[:]:
            agent.battery_max = agent.battery_soc
            agent.load_demand = big_data_file[self.steps][agent.id][0] #3*np.random.random_sample()
            agent.gen_output = big_data_file[self.steps][agent.id][1] #3*np.random.random_sample() + 2
            self.load_demand_list[agent.id] = agent.load_demand
            self.gen_output_list[agent.id] = agent.gen_output

            agent.gen_output_max = agent.gen_output
            agent.constraint_min = 0
            agent.constraint_max = 0.01 + min(agent.gen_output_max + agent.battery_ramp, agent.gen_output_max + max(agent.battery_soc, 0))
            self.P_supply_list[agent.id] = agent.constraint_max
            self.P_demand_list[agent.id] = agent.load_demand

        for agent in self.agents[:]:
            self.alpha_vector[agent.id] = agent.alpha
            self.beta_vector[agent.id] = agent.beta
            self.gamma_vector[agent.id] = agent.gamma

        """ PSO functions """
        # self.inertia_weight = self.inertia_max - ((self.inertia_max - self.inertia_min)/self.iter_max) * self.iteration
        self.inertia_weight = 0.5
        # P_pbc = 1 + (sum(self.P_i) - P_d - P_l)**2
        # F_cost = 1 + abs(sum(self.costfunction_list) - self.F_min) / (self.F_max - self.F_min)
        # eval_function = 1/(F_cost + P_pbc)

        def costfunction(x, *args):
            """ Costfunction """
            """ agents specific weights"""
            alpha_vector_pso = args[0]
            beta_vector_pso = args[1]
            gamma_vector_pso = args[2]

            return sum(alpha_vector_pso) + np.dot(beta_vector_pso, x) + np.dot(gamma_vector_pso, x**2)


        def Buyer_objectivefunction(x, *args_buyer):
            """ Costfunction """
            """ agents specific weights"""
            c_i = x
            E_i_opt = args[0]
            c_l_opt = args[1]
            E_j_opt = args[2]
            c_l_opt = args[3]

            lambda11 = args[4]
            lambda12= args[5]

            return  (abs(E_i_opt - E_j_opt * c_i / (c_l_opt + c_i))) ** lambda11 + (c_i * E_j_opt * (c_i / (c_l_opt + c_i))) ** lambda12

        def Seller_objectivefunction(x, *args_seller):
            """ Costfunction """
            """ agents specific weights"""
            alpha_vector_pso = args[0]
            beta_vector_pso = args[1]
            gamma_vector_pso = args[2]

            return - ( (R_p_opt * (E_j_p_opt * (1 - w) / (E_p_opt + E_j_p_opt * (1 - w)))) ** lambda21
                       + (R_d_opt * (E_j_opt * w / (E_d_opt + E_j_opt * w))) ** lambda22 )





        def constraints(x, *args):
            """ inequality constraints"""
            """ Power balance; sum of all produced energy is equal to its demand"""
            P_demand_pso = args[3]
            # N = args[4]
            # P_max = args[5]
            # P_min = args[6]
            # P_prev = args[7]
            # P_ramp = args[8]
            #
            # """ generator operation constraints """
            # constraint_max_power = np.zeros(N)
            # constraint_min_power = np.zeros(N)
            #
            # for i in range(N):
            #     constraint_max_power[i] = min(P_max[i], P_prev[i] + P_ramp[i]) - x[i]
            #     constraint_min_power[i] = x[i] - max(P_min[i], P_prev[i] - P_ramp[i])

            return  0.05 - abs((sum(x) - P_demand_pso)) #, constraint_max_power, constraint_min_power]




        """ close the gap by reducing load"""
        # gap = sum(self.P_supply_list) - sum(self.P_demand_list)
        # if gap < 0:
        #     deficit = -(gap)
        #     suplus = 0
        # elif gap > 0:
        #     surplus = gap
        #     deficit = 0

        # P_demand = sum(self.P_demand_list) - deficit

        """ close the gap by increasing production by including battery support """
        P_demand = sum(self.P_demand_list)
        print('total demand,', P_demand)
        print('total supply,', sum(self.P_supply_list))

        alpha_vector = self.alpha_vector
        beta_vector = self.beta_vector
        gamma_vector = self.gamma_vector

        P_max = np.zeros(N)
        P_min = np.zeros(N)
        P_prev = np.zeros(N)
        P_ramp = np.zeros(N)

        for agent in self.agents[:]:
            P_max[agent.id] = agent.gen_output
            P_prev[agent.id] = agent.P_prev
            P_ramp[agent.id] = agent.ramp_max

        args = [alpha_vector, beta_vector, gamma_vector, P_demand, N, P_max, P_min, P_prev, P_ramp]
        lb = np.zeros(N)
        ub = np.ones(N)

        """ upper/lower bounds """
        for agent in self.agents[:]:
            lb[agent.id] = agent.constraint_min
            ub[agent.id] = agent.constraint_max  # + agent.battery_ramp



        x_opt, f_opt = pso(costfunction, lb, ub, ieqcons=[constraints], args=args, swarmsize = 1000, omega = 0.9, phip = 0.6, phig = 0.6, maxiter = 1000, minstep=1e-1, minfunc=1e-1) #
        results = x_opt

        """ run pyswarm PSO"""
        iteration = 0
        # while True:
        #     """ pyswarm"""
        #     print(x_opt)
        #     iteration += 1
        #     if abs(sum(x_opt) - sum(self.P_demand_list)) < 0.1 or iteration > 3:
        #         break
        #     else:
        #         pass
        # results = x_opt

        """ run pyswarmS PSO"""
        # options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        # optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)
        # cost, pos = optimizer.optimize(costfunction, print_step=100, iters=1000, verbose=3)
        # results = cost, pos


        print(results)

        for agent in self.agents[:]:
            self.P_optimized_list[agent.id] = results[agent.id]
            agent.battery_soc += - x_opt[agent.id] + agent.gen_output - agent.load_demand
            agent.P_prev = results[agent.id]
            self.battery_soc_list[agent.id] = agent.battery_soc
        self.steps += 1
        self.time += 1

        return results, self.P_supply_list, self.P_demand_list, self.gen_output_list, self.load_demand_list, self.battery_soc_list


class DataAgent_PSO(Agent):
    def __init__(self, model):
        super().__init__(self, model)

        # self.results
        # self.P_supply_list
        # self.P_demand_list
        # self.gen_output_list
        # self.load_demand_list
        #
