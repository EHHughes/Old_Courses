# Create a fast Bifurcation Diagram (stable points only) for the sine map
import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
from sympy import sin,pi
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern"]})

x = sy.symbols('x')
r = sy.symbols('r')

f = 0.25*r*sin(pi*x)
f_low_r = f.subs(r,1)
f_mid_r = f.subs(r,2.8)
f2_mid_r = f.subs([(x,f),(r,2.8)])
f2_high_r = f.subs([(x,f),(r,3.2)])

x_vals = np.linspace(0,1,100)
f_vals = np.empty((100,2))
f2_vals = np.empty((100,2))
for i in range(0,100):
    f_vals[i,0] = f_low_r.subs(x,x_vals[i])
    f_vals[i,1] = f_mid_r.subs(x,x_vals[i])
    f2_vals[i,0] = f2_mid_r.subs(x,x_vals[i])
    f2_vals[i,1] = f2_high_r.subs(x,x_vals[i])



fig1, (ax1, ax2) = plt.subplots(1,2)
fig1.tight_layout()
ax1.grid(True)
ax1.set_xlabel('$\displaystyle x_n$', labelpad = -2, loc = 'right', size = 14)
ax1.set_ylabel('$\displaystyle x_{n+1}$', labelpad = -2, loc = 'top', rotation = 'horizontal', size = 14)
ax1.plot(x_vals,f_vals[:,0])
ax1.plot(x_vals,x_vals)

ax2.grid(True)
ax2.set_xlabel('$\displaystyle x_n$', labelpad = -2, loc = 'right', size = 14)
ax2.set_ylabel('$\displaystyle x_{n+1}$', labelpad = -2, loc = 'top', rotation = 'horizontal', size = 14)
ax2.plot(x_vals,f_vals[:,1])
ax2.plot(x_vals,x_vals)

fig2, (ax3,ax4) = plt.subplots(1,2)
fig2.tight_layout()
ax3.grid(True)
ax3.set_xlabel('$\displaystyle x_n$', labelpad = -2, loc = 'right', size = 14)
ax3.set_ylabel('$\displaystyle x_{n+2}$', labelpad = -2, loc = 'top', rotation = 'horizontal', size = 14)
ax3.plot(x_vals,f2_vals[:,0])
ax3.plot(x_vals,x_vals)

ax4.grid(True)
ax4.set_xlabel('$\displaystyle x_n$', labelpad = -2, loc = 'right', size = 14)
ax4.set_ylabel('$\displaystyle x_{n+2}$', labelpad = -2, loc = 'top', rotation = 'horizontal', size = 14)
ax4.plot(x_vals,f2_vals[:,1])
ax4.plot(x_vals,x_vals)

plt.show()
