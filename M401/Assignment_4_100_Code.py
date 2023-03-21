# plot parameters etc
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

# histogram generator
def hist_gen(x0,shift,length):
    # create the map
    def q100_func(x,shift):
        return np.modf(x+shift)[0]
    
    # create the 10^4/5/6 orbits function
    orbits = []
    weights = []
    for k in length:
        # create new orbit 
        orbit = np.empty((10**k,))
        x_new = x0
        for i in range(1,10**k):
            x_new = q100_func(x_new,shift)
            orbit[i] = x_new
        orbits += [orbit]
        weights += [np.ones_like(orbit) / len(orbit)]

    # plot the output ax1
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.hist(orbits[0], bins = 100, range = (0,1), weights = weights[0])
    ax1.set_xlim(0,1)
    ax1.set_xlabel('$x$', loc = 'right', fontsize = 18)
    ax1.set_ylabel('Share', loc = 'top', rotation = 'horizontal', fontsize = 18)
    ax1.yaxis.labelpad = 0
    ax1.tick_params(labelsize = 14)
    # do the same for x2
    ax2.hist(orbits[1], bins = 100, range = (0,1), weights = weights[1])
    ax2.set_xlim(0,1)
    ax2.set_xlabel('$x$', loc = 'right', fontsize = 18)
    ax2.set_ylabel('Share', loc = 'top', rotation = 'horizontal', fontsize = 18)
    ax2.yaxis.labelpad = 0
    ax2.tick_params(labelsize = 14)
    # do x3
    ax3.hist(orbits[2], bins = 100, range = (0,1), weights = weights[2])
    ax3.set_xlim(0,1)
    ax3.set_xlabel('$x$', loc = 'right', fontsize = 18)
    ax3.set_ylabel('Share', loc = 'top', rotation = 'horizontal', fontsize = 18)
    ax3.yaxis.labelpad = 0
    ax3.tick_params(labelsize = 14)

    # tweak padding
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.6, hspace=0.5)

    # return our figure
    return(fig)

# create the required figures and show them
shifts = [2/3,21/34,0.5*(np.sqrt(5)-1),1/np.sqrt(3)]
figs = []
for s in shifts:
    figs += [hist_gen(0,s,(4,5,6))]
plt.show()
