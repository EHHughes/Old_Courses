# Import what we need
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern"]})

# Write up the fixed points as inputs
fixed_point_neg = (-7 - np.sqrt(609))/28
fixed_point_pos = (-7 + np.sqrt(609))/28
print(fixed_point_neg)
print(fixed_point_pos)

# Get the Jacobian for each point
fixed_point_neg_mat = np.array([[-2.8*fixed_point_neg, 1],[0.3,0]])
fixed_point_pos_mat = np.array([[-2.8*fixed_point_pos, 1],[0.3,0]])

# Get the eigenvalues and vectors
eig_neg = np.linalg.eig(fixed_point_neg_mat)
eig_pos = np.linalg.eig(fixed_point_pos_mat)
print(eig_neg)
print(eig_pos)

# Get the a_1 term
a_1_neg = eig_neg[1][1][0]/eig_neg[1][0][0]
a_1_pos = eig_pos[1][1][0]/eig_pos[1][0][0]
print(a_1_neg)
print(a_1_pos)

# Print checks for each term
print(a_1_neg*(a_1_neg - 2.8*fixed_point_neg) - 0.3)
print(a_1_pos*(a_1_pos - 2.8*fixed_point_pos) - 0.3)

# Get the a_2 term
a_2_neg = a_1_neg*1.4/((-2.8*fixed_point_neg + a_1_neg)**2 + a_1_neg)
a_2_pos = a_1_pos*1.4/((-2.8*fixed_point_pos + a_1_pos)**2 + a_1_pos)
print(a_2_neg)
print(a_2_pos)

# Define the henon map
def henon_map(x):
    x_old = np.array(x)
    x[:,0] = 1-1.4*x_old[:,0]**2 + x_old[:,1]
    x[:,1] = 0.3*x_old[:,0]
    return(x)

# Print the test that this actually works
print(np.array([[fixed_point_neg, 0.3*fixed_point_neg]]) == henon_map(np.array([[fixed_point_neg, 0.3*fixed_point_neg]])))

# Generate our approximations of the unstable manifold about each fixed point
num_points = 2000
off_set_neg = np.linspace(0,0.05, num_points)
off_set_pos = np.linspace(0.05,0.05, num_points)
neg_off_set_points_x = (off_set_neg + fixed_point_neg).T
neg_off_set_points_y = (a_1_neg*off_set_neg + a_2_neg*off_set_neg**2 +0.3*fixed_point_neg).T
neg_set_points = np.column_stack((neg_off_set_points_x, neg_off_set_points_y))
pos_off_set_points_x = (off_set_pos + fixed_point_pos).T
pos_off_set_points_y = (a_1_pos*off_set_pos + a_2_pos*off_set_pos**2 +0.3*fixed_point_pos).T
pos_set_points = np.column_stack((pos_off_set_points_x, pos_off_set_points_y))
set_points = np.concatenate((neg_set_points,pos_set_points), axis = 0)
orbits = 50

# Create a matrix to holds the orbits onwards
out_orbit = np.empty((2*num_points*orbits,2))

# Iterate forward the maps to create the orbits
for i in range(orbits):
    out_orbit[i*2*num_points:i*2*num_points + 2*num_points,:] = henon_map(set_points)

# Plot everything
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(out_orbit[:,0], out_orbit[:,1], s = 0.005)
ax.set_xlabel('$y$', loc = 'right', fontsize = 18)
ax.set_ylabel('$x$', loc = 'top', rotation = 'horizontal', fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.show()

