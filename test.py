import numpy as np
from scipy.optimize import minimize


x0 = [1,5,5,1] # initial guess

"""Declaration of global constants"""
global x2_fixed
global x3_fixed
x2_fixed = 3
x3_fixed = 1.2


def objective(x, sign=-1.0):            # SIGN= -1 gives maximization instead of minimization
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    return sign*x1*x4*(x1 + x2 + x3) + x3


def constraint1(x):
    return x[0]*x[1]*x[2]*x[3] - 25.0


def constraint2(x):
    sum_sq = 40
    return sum_sq - x[0]**2 - x[1]**2 -x[2]**2 -x[3]**2


def constraintx2(x):
    return x2_fixed - x[1]
# print(objective(x0))

def constraintx3(x):
    return x3_fixed - x[2]



b = (1.0, 5.0)
bnds = (b,b,b,b)
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'eq', 'fun': constraint2}

"""fix parameters"""
con3 = {'type': 'eq', 'fun': constraintx2}
con4 = {'type': 'eq', 'fun': constraintx3}



cons =[con1,con2,con3,con4]


sol = minimize(objective, x0, method='SLSQP', bounds= bnds,constraints = cons)




print(sol)

