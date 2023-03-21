# Import the relevant python modules
import sympy as sy
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

# Define our variables we will be working with
x = sy.symbols('x')

# Create an expression for f
f = 3.835*x*(1-x)

# Nest to create f3
f2 = f.subs(x, f)
f3 = f2.subs(x,f)

# (a) 
# Iterate forwards to find a_3 and a_6 (the points the 
# ends of our range of interest get mapped to)
a_0 = 0.15
a_3 = f3.subs(x,a_0)
a_6 = f3.subs(x,a_3)

# Print off to show that the endpoints of 
# our initial set get mapped inside the initial set

# It remains to show that the function is downward sloping 
# inside this range (so we can interpolate that all 
# the points inside [a0, a3] remain inside)

# Take the first derivative 
f3_prime = sy.diff(f3,x)


# Solve for zeros of the first derivative
f3_prime_zeros = sy.solveset(f3_prime, x)

# create 100 points across this interval to plot
x_plot_vals = np.linspace(0.15,0.1533,100)
f3_vals = np.empty((100,1))
for i in range(0,100):
    f3_vals[i] = f3.subs(x,x_plot_vals[i])

# Draw a picture to show that f^3 is monotonically decreasing on this range
fig1 = plt.figure()
ax = fig1.add_subplot()
ax.grid(True)
ax.set_xlabel('$\displaystyle x$', labelpad = -10, loc = 'right', fontsize = 14)
ax.set_ylabel('$\displaystyle f^3(x)$', labelpad = -15, loc = 'top', rotation = 'horizontal', fontsize = 14)
ax.plot(x_plot_vals,f3_vals)
plt.tight_layout()


# Show the derivative at the left hand side of this is negative 
f3_prime_a_0 = f3_prime.subs(x,a_0)

# So the initial derivative is negative. Thus f^3 maps 
# the interval [a0,a3] to itself and consequently there must 
# be a period three point inside this interval

# (b) This is obvious from above

# (c)
# We need to repeat the proceedure above to show that 
# f3' is bounded on this set
# Evaluate the first derivative at the endpoints of this 
# domain
f3_prime_a_3 = f3_prime.subs(x,a_3)
print([f3_prime_a_0,f3_prime_a_3])

# find the second derivative of f3
f3_two_prime = sy.diff(f3_prime,x)

# Solve for zeros
f3_two_prime_zeros = sy.solveset(f3_two_prime,x)

# So the derivative is monotonically increasing between 
# the two endpoints (as there are no changes in concavity 
# along this interval) and is bounded at both endpoints 
# so it fulfills the condition in (c). Consequently we 
# can safely assume that the period-3 point is attracting
print(f3_two_prime_zeros)

f3_prime_vals = np.empty((100,1))
for i in range(0,100):
    f3_prime_vals[i] = f3_prime.subs(x,x_plot_vals[i])

fig2 = plt.figure()
ax = fig2.add_subplot()
ax.grid(True)
ax.set_xlabel('$\displaystyle x$', labelpad = -10, loc = 'right', fontsize = 14)
ax.set_ylabel('$\displaystyle (f^3)\'$', labelpad = -15, loc = 'top', rotation = 'horizontal', fontsize = 14)
ax.plot(x_plot_vals,f3_prime_vals)
plt.tight_layout()


# (d)
# Use lambdify to convert our function into something that 
# can be looped quickly.
f_lambda = sy.lambdify(x,f, 'numpy')

# Loop 100 times as required
first_hundred_iters = []
x_current = a_0
for x in range(0,100):
    x_current = f_lambda(x_current)
    first_hundred_iters += [f_lambda(x_current)]

# Print the forward first 100 terms of the forward orbit

# Yes, yes it is. Plot to demonstrate
fig3 = plt.figure()
ax = fig3.add_subplot()
ax.grid(False)
ax.set_xlabel('$f^n(a_0)$', loc = 'center', fontsize = 14)
ax.hist(first_hundred_iters, bins = 100)
plt.tight_layout()
plt.show()

