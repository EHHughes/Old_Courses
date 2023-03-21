# Create a fast Bifurcation Diagram (stable points only) for the sine map
from cmath import nan
import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
from math import pi
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern"]})


# Initial values to feed in
r_vals = np.linspace(0,4,6000)
r_vals = matlib.repmat(r_vals, 100,1)


# Sine map function
def sine_map(x,r):
    x_nplusone = (r/4)*np.sin(pi*x)
    return x_nplusone


# Sine log function
def log_map(x,r):
    x_nplusone = r*x*(1-x)
    return x_nplusone

# Create a matrix to hold our output data
x_n_sine = np.linspace(0,1,100)
x_n_sine = matlib.repmat(x_n_sine,6000,1).T
x_n_log = x_n_sine


# wrap a for loop and try for 10000 times
for i in range(0,2000):
    x_n_sine= sine_map(x_n_sine,r_vals)
    x_n_log = log_map(x_n_log,r_vals)


# Remove the excess points from the plot
x_n_sine[:5,r_vals[1,:] > 4/pi] = np.nan
x_n_log[(x_n_log < 0.01) & (r_vals > 1.01)] = np.nan
x_n_log[(x_n_log < 0.00001) & (r_vals > 1)] = np.nan


# Plotting
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.scatter(r_vals,x_n_log,s=0.01, alpha = 0.5)
ax1.set_xlabel('$r$', loc = 'right', fontsize = 18)
ax1.set_ylabel('$x$', loc = 'top', rotation = 'horizontal', fontsize = 18)
ax1.tick_params(axis = 'x', labelsize = 14)
ax1.tick_params(axis = 'y', labelsize = 14)

ax2.scatter(r_vals,x_n_sine,s=0.01, alpha = 0.5)
ax2.set_xlabel('$r$', loc = 'right', fontsize = 18)
ax2.set_ylabel('$x$', loc = 'top', rotation = 'horizontal', fontsize = 18)
ax1.tick_params(axis = 'x', labelsize = 14)
ax1.tick_params(axis = 'y', labelsize = 14)

plt.tight_layout(h_pad = 0.4)
plt.show()