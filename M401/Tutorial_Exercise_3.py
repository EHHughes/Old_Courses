import numpy as np
from matplotlib import pyplot as plt

out_vals = np.empty((10000,2))
out_vals[1,:] = [0,0]

for i in range(1,10000):
    out_vals[i,:] = [1+out_vals[i-1,1]-1.4*out_vals[i-1,0]**2,0.3*out_vals[i-1,0]]

plt.scatter(out_vals[:,0],out_vals[:,1])
plt.show()