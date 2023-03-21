# Import the relevant python modules
import sympy as sy

# Define our variables we will be working with
x, a = sy.symbols('x, a')

# Create an expression for f using x and a 
f = x-2*x**3 # INSERT REAL EXPRESSION HERE

# Nest to create f3
f2 = f.subs(x, f)
print(f2)

# fixed point finder
fix2 = f2 -x
print(fix2)

# Take the first derivative 
f2_prime = sy.diff(f2,x)

# Solve for zeros of various functions first derivative
fix2_zeros = sy.solveset(fix2, x)
print(fix2_zeros)