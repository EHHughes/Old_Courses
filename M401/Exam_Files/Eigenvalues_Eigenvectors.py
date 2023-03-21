# get the files we need
import sympy as sy

x,y,mu = sy.symbols('x,y,mu')

# get the x and y directions
fx = y # xdot function
fy = mu*y -x*(x-1/3)*(1-x) # ydot function
fxx = sy.diff(fx,x)
fxy = sy.diff(fx,y)
fyx = sy.diff(fy,x)
fyy= sy.diff(fy,y)
print(fx)
print(sy.diff(fx,x))
# bind to a vector valued function
jac = sy.Matrix([[fxx,fxy],[fyx,fyy]])

jac0 = jac.subs({x:0,y:0})
print(jac0)

# get the eigenvalues and eigenvectors
eigs = jac0.eigenvects()
print(eigs)