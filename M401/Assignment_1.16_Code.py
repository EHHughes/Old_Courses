# Import the relevant python modules
import sympy as sy
import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern"]})

# Initial values to feed in
r_vals = np.linspace(3.7,3.9,700)
r_vals = matlib.repmat(r_vals, 200,1)

# Sine map function
def log_map(x,r):
    x_nplusone = r*x*(1-x)
    return x_nplusone

log_map_vectorized = np.vectorize(log_map)

# Generate linearly spaced points to iterate from
x_n = np.linspace(0.1,1,200)
x_n = matlib.repmat(x_n,700,1).T
for i in range(0,400):
    x_n = log_map_vectorized(x_n,r_vals)

x_n_final = x_n
x_n_final = x_n_final[x_n[:,0] > np.ones(200)*0.2, :]
r_vals = r_vals[x_n[:,0] > np.ones(200)*0.2, :]

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(r_vals,x_n_final,s=0.005, alpha = 0.9)
ax.set_xlabel('$r$', loc = 'right', fontsize = 14)
ax.set_ylabel('$x$', loc = 'top', rotation = 'horizontal', fontsize = 14)
plt.tight_layout()
plt.show()
