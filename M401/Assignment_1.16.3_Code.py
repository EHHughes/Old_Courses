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

# Substitute for the first r value of interest
f3_low_r = f3.subs(r,3.7)
f3_mid_r = f3.subs(r,3.835)
f3_high_r = f3.subs(r,3.86)

# Create fixed value finders for each of these r values
f3_low_r_fixed = f3_low_r - x 
f3_mid_r_fixed = f3_mid_r - x
f3_high_r_fixed = f3_high_r - x

# Use nsolve to go looking for fixed points, based on the eyeballs we have from the figure
roots = []
root_1 = sy.nsolve(f3_mid_r_fixed, 0.16)
roots += [root_1]

root_2 = sy.nsolve(f3_mid_r_fixed,0.15)
roots += [root_2]

root_3 = sy.nsolve(f3_mid_r_fixed,0.4)
roots += [root_3]

root_4 = sy.nsolve(f3_mid_r_fixed,0.3)
roots += [root_4]

root_5 = sy.nsolve(f3_mid_r_fixed,0.2)
roots += [root_5]

root_6 = sy.nsolve(f3_mid_r_fixed,0.99)
roots += [root_6]

# Find f3_prime and compute at each of these hypothesized roots. 
f3_prime = sy.diff(f3_mid_r,x)
for i in roots:
    df3 = f3_prime.subs(x,i)
    print(df3)
print(roots)