import numpy as np


a = [0,1,2,3,4,5,6,7,8,9]



new_step_time = 4

print(len(a)/new_step_time)

b = np.zeros(int(len(a)/new_step_time))

for step in range(int(len(a)/new_step_time)):
    b[step] = sum(a[new_step_time*step:(new_step_time*step + new_step_time)])

print(b)