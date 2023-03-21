# Plot T and T^2 for each of the S values
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern"]})

def cubic_map(x,r):
    x_n1 = r*x - x**3
    return x_n1

cubic_map_vec = np.vectorize(cubic_map)

# Create x and r_vals for plotting
low_r = 0.5
mid_r = 1
high_r = 1.5
x_vals_low_r = np.linspace(-sqrt(low_r+1),sqrt(low_r+1),200)
x_vals_mid_r = np.linspace(-sqrt(mid_r+1),sqrt(mid_r+1),200)
x_vals_high_r = np.linspace(-sqrt(high_r+1),sqrt(high_r+1),200)

r_vals = np.empty((200,3))
r_vals[:,0] = 0.5*np.ones(200)
r_vals[:,1] = np.ones(200)
r_vals[:,2] = 1.5*np.ones(200)

f_vals = np.empty((200,3))

# Generate f_vals for our functions
f_vals[:,0] = cubic_map_vec(x_vals_low_r,r_vals[:,0])
f_vals[:,1] = cubic_map_vec(x_vals_mid_r,r_vals[:,1])
f_vals[:,2] = cubic_map_vec(x_vals_high_r,r_vals[:,2])

fig1, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.grid(True)
ax1.set_xlim([-1.8,1.8])
ax1.set_ylim([-2.1,2.1])
ax1.set_xlabel('$\displaystyle x_n$', labelpad = -2, loc = 'right', size = 14)
ax1.set_ylabel('$\displaystyle x_{n+1}$', labelpad = -10, loc = 'top', rotation = 'horizontal', size = 14)
ax1.plot(x_vals_low_r,f_vals[:,0])
ax1.plot(x_vals_low_r,x_vals_low_r)

ax2.grid(True)
ax2.set_xlim([-1.8,1.8])
ax2.set_ylim([-2.1,2.1])
ax2.set_xlabel('$\displaystyle x_n$', labelpad = -2, loc = 'right', size = 14)
ax2.set_ylabel('$\displaystyle x_{n+1}$', labelpad = -10, loc = 'top', rotation = 'horizontal', size = 14)
ax2.plot(x_vals_mid_r,f_vals[:,1])
ax2.plot(x_vals_mid_r,x_vals_mid_r)

ax3.grid(True)
ax3.set_xlim([-1.8,1.8])
ax3.set_ylim([-2.1,2.1])
ax3.set_xlabel('$\displaystyle x_n$', labelpad = -2, loc = 'right', size = 14)
ax3.set_ylabel('$\displaystyle x_{n+1}$', labelpad = -10, loc = 'top', rotation = 'horizontal', size = 14)
ax3.plot(x_vals_high_r,f_vals[:,2])
ax3.plot(x_vals_high_r,x_vals_high_r)

plt.tight_layout(h_pad = 0.4)
plt.show()