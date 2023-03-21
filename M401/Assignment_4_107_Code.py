import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

# this is the operator for Q100
def q100_func(x,shift):
    return np.modf(x+shift)[0] 

# this function is the transformed function for Q102 on (0,1]
def q102_func(x,params):
    return np.sqrt(x)

def ulam_method(func,bins,K,params):
    # take as input a function defined on [0,1] the number of bins and the number of points 
    # inside each bin

    # build an array where each row is all the K points in some bin
    x_in_bins = []
    for i in range(bins.shape[0]-1):
        x_in_bins += [np.linspace(bins[i],bins[i+1],K)]
    x_in_bins = np.array(x_in_bins)

    # apply the function of interest to the array
    x_trans = func(x_in_bins,params)

    # Create the output array 
    Ulam_matrix = np.empty((bins.shape[0]-1,bins.shape[0]-1))

    # create each entry in the Ulam matrix
    for j in range(x_in_bins.shape[0]):
        Ulam_matrix[:,j] = np.histogram(x_trans[j,:],bins,density=False)[0]/K

    # return the Ulam matrix and the 
    return Ulam_matrix

# draw some pictures for mass transport starting on 0,0.1 for each angle in Q100
angles = [2/3,21/34,(np.sqrt(5)-1)/2,1/np.sqrt(3)]
figs = []
for angle in angles:
    mat_Q100 = ulam_method(q100_func,np.linspace(0,1,1001),100,angle)
    fig, axes = plt.subplots(2,3,subplot_kw={'projection': 'polar'})
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.4, hspace=0.5)
    figs += [fig]
    axes = list(np.reshape(axes,(6,)))
    x_axis = np.linspace(0,1,1000)
    border = 0.85*np.reshape(np.ones((1,1000)),(1000,))
    blob = np.zeros(1000)
    blob[0:100] = 0.1
    counter = 0
    x_reg = x_axis[blob > 0.05]
    axes[0].scatter(2*np.pi*x_reg, np.ones(x_reg.shape[0]))
    axes[0].plot(2*np.pi*x_axis, border,color='black')
    axes[0].set_rticks([])
    axes[0].set_ylim([-1.2,1.2])
    axes[0].set_xticks(np.linspace(0,2*np.pi,9))
    axes[0].set_xticklabels(['0','1/8','1/4','3/8','1/2','5/8','3/4','7/8',''])
    axes[0].tick_params(length = 0, grid_alpha = 0)
    axes[0].spines['polar'].set_visible(False)
    axes[0].set_title('Iteration Number ' + str(counter), va = 'bottom', fontsize = 18)
    for ax in axes[1:]:
        counter += 1
        blob= np.matmul(mat_Q100,blob)
        x_reg = x_axis[blob > 0.05]
        ax.scatter(2*np.pi*x_reg, np.ones(x_reg.shape[0]))
        ax.plot(2*np.pi*x_axis, border,color='black')
        ax.set_rticks([])
        ax.set_ylim([-1.2,1.2])
        ax.set_xticks(np.linspace(0,2*np.pi,9))
        ax.set_xticklabels(['0','1/8','1/4','3/8','1/2','5/8','3/4','7/8',''])
        ax.tick_params(length = 0, grid_alpha = 0)
        ax.spines['polar'].set_visible(False)
        ax.set_title('Iteration Number ' + str(counter), va = 'bottom', fontsize = 18)

# get the Ulam matrix for Q102
max_eigvec = []
for k in [10,100]:
    mat_Q102 = ulam_method(q102_func,np.linspace(0,2,101),k,0)
    # from this grab the largest eigenvalue and eigenvector
    eigvals_102, eigvec_102 = eig(mat_Q102)
    max_eigval = eigvals_102[np.argmax(np.abs(eigvals_102))]
    max_eigvec += [eigvec_102[:,np.argmax(np.abs(eigvals_102))]]

# use the matrix we have just computed to draw some pictures of mass transport for a blob starting on 0,0.1 
fig, axes = plt.subplots(3,2)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.7)
axes = list(np.reshape(axes,6))
x_axis = np.linspace(0,2,mat_Q102.shape[0])
blob = 0.5*np.ones(mat_Q102.shape[0])
axes[0].plot(x_axis,blob)
axes[0].set_xlim(0,2)
axes[0].set_xlabel('$x$', loc = 'right', fontsize = 18)
axes[0].set_ylabel('Mass', loc = 'top', rotation = 'horizontal', fontsize = 18)
axes[0].yaxis.labelpad = 0
axes[0].tick_params(labelsize = 14)
for ax in axes[1:]:
    blob = np.matmul(mat_Q102,blob)
    ax.plot(x_axis,blob)
    ax.set_xlim(0,2)
    ax.set_xlabel('$x$', loc = 'right', fontsize = 18)
    ax.set_ylabel('Mass', loc = 'top', rotation = 'horizontal', fontsize = 18)
    ax.yaxis.labelpad = 0
    ax.tick_params(labelsize = 14)

# Also plot the eigenvector in a separate figure
fig2, ax_eig = plt.subplots(1,1)
ax_eig.plot(np.linspace(0,2,max_eigvec[0].shape[0]),max_eigvec[0]) # plot the eigenvector distribution
ax_eig.set_xlim(0,2)
ax_eig.set_xlabel('$x$', loc = 'right', fontsize = 18)
ax_eig.set_ylabel('Mass', loc = 'top', rotation = 'horizontal', fontsize = 18)
ax_eig.yaxis.labelpad = 0
ax_eig.tick_params(labelsize = 14)
plt.show()