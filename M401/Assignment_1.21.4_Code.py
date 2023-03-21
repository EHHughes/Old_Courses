# Plot T and T^2 for each of the  s= \root 2 values
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern"]})

def tent_map(x,s):
    if x < 0.5:
        x_n1 = s*x
    else:
        x_n1 = s*(1-x)
    return x_n1

def tent_map_2(x,s):
    if x < 0.5:
        x_n1 = s*x
    else:
        x_n1 = s*(1-x)
    if x_n1 < 0.5:
        x_n2 = s*x_n1
    else:
        x_n2 = s*(1-x_n1)
    return x_n2

tent_map_1_vec = np.vectorize(tent_map)
tent_map_2_vec = np.vectorize(tent_map_2)

x_vals = np.linspace(0,1,200)
r_low = 1.3*np.ones(200)
f_vals = np.empty((200,3))
f2_vals = np.empty((200,3))

f_vals[:,0] = tent_map_1_vec(x_vals,r_low)
f2_vals[:,0] = tent_map_2_vec(x_vals,r_low)

ax1 = plt.subplot()
ax1.grid(True)
ax1.set_xlabel('$\displaystyle x_n$', labelpad = -2, loc = 'right', size = 14)
ax1.set_ylabel('$\displaystyle x_{n+1}$', labelpad = -10, loc = 'top', rotation = 'horizontal', size = 14)
ax1.plot(x_vals,f_vals[:,0])
ax1.plot(x_vals,x_vals)


plt.show()