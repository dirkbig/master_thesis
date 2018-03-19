""" Particle swarm optimization """
from functions.function_file import *
from functions.pso_custom import *
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


        self.battery_capacity_n = 15
        self.battery_ramp = 1
        self.soc_actual = 0.5 * self.battery_capacity_n
        self.battery_max = self.battery_capacity_n

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

        self.soc_influx = 0

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
        self.big_data_file = big_data_file
        self.soc_actual_list = np.zeros(N)

    def step(self, N, lambda_set):
        print('step ',self.steps)

        lambda_set_notused = lambda_set
        """ update values from data """
        for agent in self.agents[:]:
            agent.battery_max = agent.battery_capacity_n
            agent.load_demand = self.big_data_file[self.steps][agent.id][0] #3*np.random.random_sample()
            agent.gen_output = self.big_data_file[self.steps][agent.id][1] #3*np.random.random_sample() + 2
            self.load_demand_list[agent.id] = agent.load_demand
            self.gen_output_list[agent.id] = agent.gen_output

            agent.gen_output_max = agent.gen_output
            agent.constraint_min = 0
            agent.constraint_max = 0.01 + min(agent.gen_output_max + agent.battery_ramp, agent.gen_output_max + max(agent.battery_capacity_n, 0))
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
            cost = sum(alpha_vector_pso) + np.dot(beta_vector_pso, x) + np.dot(gamma_vector_pso, x**2)
            return cost

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

        def Seller_objectivefunction(w, *args_seller):
            """ Costfunction """
            """ agents specific weights"""
            R_p_opt = args_seller[0]
            E_j_p_opt = args_seller[1]
            E_p_opt = args_seller[2]
            R_d_opt = args_seller[3]
            E_j_opt = args_seller[4]
            E_d_opt = args_seller[5]

            lambda21 = args_seller[6]
            lambda22 = args_seller[7]

            return - ( (R_p_opt * (E_j_p_opt * (1 - w) / (E_p_opt + E_j_p_opt * (1 - w)))) ** lambda21
                       + (R_d_opt * (E_j_opt * w / (E_d_opt + E_j_opt * w))) ** lambda22 )




        def constraints(x, *args):
            """ inequality constraint,
                Power balance; sum of all produced energy is equal to its demand to a certain extent,
                batteries can ofcourse cope with a bit of inequality"""
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

            return  1 - abs((sum(x) - P_demand_pso)) #, constraint_max_power, constraint_min_power]




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
        P_supply = sum(self.P_supply_list)

        if P_demand > P_supply:
            P_deficit = P_supply - P_demand
            P_demand = P_supply


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

        max_it = 1000
        x_opt, f_opt, list_on_iteration, it = pso_custom(costfunction, lb, ub, ieqcons=[constraints], args=args, swarmsize = 2000, omega = 0.9, phip = 0.6, phig = 0.6, maxiter = max_it, minstep=1e-0, minfunc=1e-0) #
        results = x_opt
        if it > max_it - 1:
            x_opt = np.zeros(N)



        for agent in self.agents[:]:
            agent.soc_influx = 0
            agent.deficit = 0
            self.P_optimized_list[agent.id] = results[agent.id]
            agent.P_prev = results[agent.id]

            agent.soc_influx = - x_opt[agent.id] + agent.gen_output - agent.load_demand
            if agent.soc_actual + agent.soc_influx < 0:
                """ battery will deplete """
                agent.deficit = abs(agent.soc_actual + agent.soc_influx)
                agent.soc_influx = agent.E_i_demand + agent.deficit
            if agent.battery_capacity_n + agent.soc_influx > agent.battery_capacity_n:
                """ battery will overflow """
                real_influx = agent.soc_influx
                agent.soc_influx = agent.battery_capacity_n - agent.soc_actual
                agent.batt_overflow = real_influx - agent.soc_influx
            self.soc_actual_list[agent] = agent.soc_actual

        self.steps += 1
        self.time += 1

        return results, self.P_supply_list, self.P_demand_list, self.gen_output_list, self.load_demand_list, self.soc_actual_list


class DataAgent_PSO(Agent):
    def __init__(self, model):
        super().__init__(self, model)

