# Import the relevant python modules
import sympy as sy
import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern"]})

# Define our variables we will be working with
x,r = sy.symbols('x r')

# Create an expression for f
f = r*x*(1-x)

# Nest to create f3
f2 = f.subs(x, f)
f3 = f2.subs(x,f)

# Nest Again to Create f6
f6 = f3.subs(x,f3)

# Substitute for the first r value of interest
f3_low_r = f3.subs(r,3.7)
f3_mid_r = f3.subs(r,3.835)
f3_high_r = f3.subs(r,3.86)
f_low_r = f.subs(r,3.7)
f_mid_r = f.subs(r,3.835)
f_high_r = f.subs(r,3.86)
f6_low_r = f6.subs(r,3.7)
f6_mid_r = f6.subs(r,3.835)
f6_high_r = f6.subs(r,3.86)

# Create initial values 
x_plot_vals = np.linspace(0,1,200)
f3_vals = np.empty((200,3))
f6_vals = np.empty((200,3))
f_vals = np.empty((200,3))
for i in range(0,200):
    f3_vals[i,0] = f3_low_r.subs(x,x_plot_vals[i])
    f3_vals[i,1] = f3_mid_r.subs(x,x_plot_vals[i])
    f3_vals[i,2] = f3_high_r.subs(x,x_plot_vals[i])
    f6_vals[i,0] = f6_low_r.subs(x,x_plot_vals[i])
    f6_vals[i,1] = f6_mid_r.subs(x,x_plot_vals[i])
    f6_vals[i,2] = f6_high_r.subs(x,x_plot_vals[i])
    f_vals[i,0] = f_low_r.subs(x,x_plot_vals[i])
    f_vals[i,1] = f_mid_r.subs(x,x_plot_vals[i])
    f_vals[i,2] = f_high_r.subs(x,x_plot_vals[i])



fig1, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.grid(False)
ax1.set_xlabel('$x_n$', labelpad = -2, loc = 'right', fontsize = 14)
ax1.set_ylabel('$x_{n+1}$', labelpad = -10, loc = 'top', rotation = 'horizontal', fontsize = 14)
ax1.plot(x_plot_vals,f3_vals[:,0])
ax1.plot(x_plot_vals,f6_vals[:,0])
ax1.plot(x_plot_vals,x_plot_vals)
ax1.plot(x_plot_vals,f_vals[:,0])


ax2.set_xlabel('$x_n$', labelpad = -2, loc = 'right', fontsize = 14)
ax2.set_ylabel('$x_{n+1}$', labelpad = -10, loc = 'top', rotation = 'horizontal', fontsize = 14)
ax2.plot(x_plot_vals,f3_vals[:,1])
ax2.plot(x_plot_vals,f6_vals[:,1])
ax2.plot(x_plot_vals,x_plot_vals)
ax2.plot(x_plot_vals,f_vals[:,1])


ax3.set_xlabel('$x_n$', labelpad = -2, loc = 'right', fontsize = 14)
ax3.set_ylabel('$x_{n+1}$', labelpad = -10, loc = 'top', rotation = 'horizontal', fontsize = 14)
ax3.plot(x_plot_vals,f3_vals[:,2])
ax3.plot(x_plot_vals,f6_vals[:,2])
ax3.plot(x_plot_vals,x_plot_vals)
ax3.plot(x_plot_vals,f_vals[:,2])


plt.tight_layout(h_pad = 0.4)
plt.show()