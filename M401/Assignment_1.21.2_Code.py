# Plot T and T^2 for each of the S values
import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
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

# Create a matrix to hold our output data
r_vals = np.linspace(1,2,700)
r_vals = matlib.repmat(r_vals, 200,1)
x_n = np.linspace(0,1,200)
x_n = matlib.repmat(x_n,700,1).T
x_n2 = x_n

# wrap a for loop and try for 800 times
for j in range(0,1500):
    x_n = tent_map_1_vec(x_n,r_vals)

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(r_vals,x_n,s=0.005, alpha = 0.9)
ax.set_xlabel('$s$', loc = 'right', fontsize = 18)
ax.set_ylabel('$x$', loc = 'top', rotation = 'horizontal', fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.show()