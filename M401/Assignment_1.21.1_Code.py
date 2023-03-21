# Plot T and T^2 for each of the S values
import numpy as np
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

x_vals = np.linspace(0,1,200)
r_low = np.ones(200)
r_mid = 1.5*np.ones(200)
r_high = 2*np.ones(200)
f_vals = np.empty((200,3))
f2_vals = np.empty((200,3))

f_vals[:,0] = tent_map_1_vec(x_vals,r_low)
f2_vals[:,0] = tent_map_2_vec(x_vals,r_low)

f_vals[:,1] = tent_map_1_vec(x_vals,r_mid)
f2_vals[:,1] = tent_map_2_vec(x_vals,r_mid)

f_vals[:,2] = tent_map_1_vec(x_vals,r_high)
f2_vals[:,2] = tent_map_2_vec(x_vals,r_high)

fig1, (ax1, ax2, ax3) = plt.subplots(1,3)
fig2, (ax4, ax5, ax6) = plt.subplots(1,3) 
fig1.tight_layout()
fig2.tight_layout()
ax1.grid(True)
ax1.set_xlabel('$\displaystyle x_n$', labelpad = -2, loc = 'right', size = 14)
ax1.set_ylabel('$\displaystyle x_{n+1}$', labelpad = -10, loc = 'top', rotation = 'horizontal', size = 14)
ax1.plot(x_vals,f_vals[:,0])
ax1.plot(x_vals,x_vals)

ax4.grid(True)
ax4.set_xlabel('$\displaystyle x_n$', labelpad = -2, loc = 'right', size = 14)
ax4.set_ylabel('$\displaystyle x_{n+2}$', labelpad = -10, loc = 'top', rotation = 'horizontal', size = 14)
ax4.plot(x_vals,f2_vals[:,0])
ax4.plot(x_vals,x_vals)

ax2.grid(True)
ax2.set_xlabel('$\displaystyle x_n$', labelpad = -2, loc = 'right', size = 14)
ax2.set_ylabel('$\displaystyle x_{n+1}$', labelpad = -10, loc = 'top', rotation = 'horizontal', size = 14)
ax2.plot(x_vals,f_vals[:,1])
ax2.plot(x_vals,x_vals)

ax5.grid(True)
ax5.set_xlabel('$\displaystyle x_n$', labelpad = -2, loc = 'right', size = 14)
ax5.set_ylabel('$\displaystyle x_{n+2}$', labelpad = -10, loc = 'top', rotation = 'horizontal', size = 14)
ax5.plot(x_vals,f2_vals[:,1])
ax5.plot(x_vals,x_vals)

ax3.grid(True)
ax3.set_xlabel('$\displaystyle x_n$', labelpad = -2, loc = 'right', size = 14)
ax3.set_ylabel('$\displaystyle x_{n+1}$', labelpad = -10, loc = 'top', rotation = 'horizontal', size = 14)
ax3.plot(x_vals,f_vals[:,2])
ax3.plot(x_vals,x_vals)

ax6.grid(True)
ax6.set_xlabel('$\displaystyle x_n$', labelpad = -2, loc = 'right', size = 14)
ax6.set_ylabel('$\displaystyle x_{n+2}$', labelpad = -10, loc = 'top', rotation = 'horizontal', size = 14)
ax6.plot(x_vals,f2_vals[:,2])
ax6.plot(x_vals,x_vals)

plt.show()