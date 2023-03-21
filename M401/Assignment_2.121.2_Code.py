def phase_sketch(f, x_limits, y_limits, num_curves, show, plt_xlim, plt_ylim):
    # Import the packages we need and set up the plot to use latex
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import scipy.integrate as scp
    import math
    from bisect import bisect_left
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern"]})
    
    # Initlialize the plot
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.xlim(plt_xlim)
    plt.ylim(plt_ylim)
    ax.set_xlabel('$x$', loc = 'right', fontsize = 18)
    ax.set_ylabel('$y$', loc = 'top', rotation = 'horizontal', fontsize = 18)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.tight_layout()
    
    # Set a tspan and unpack x and y limits into linspace
    tspan = [0,30]
    x_inits = np.linspace(x_limits[0],x_limits[1],num=num_curves)
    y_inits = np.linspace(y_limits[0],y_limits[1],num=num_curves)

    # Create a list to hold all our solutions
    sol_list = []
    arc_length_sol = []

    # Set up initial conditions (configure as appropriate)
    inits_one = np.column_stack((x_inits,y_inits[0]*np.ones((num_curves,1))))
    inits_two = np.column_stack((x_inits[0]*np.ones((num_curves,1)),y_inits))
    inits_four = np.column_stack((x_limits[-1]*np.ones((num_curves,1)),y_inits))
    inits_extra = np.array([[1.5,3],[2,3],[3,-2],[1.9,3],[1.95,3],[1.93,3],[3,-2.5],
        [0.031,-0.021],[-0.031,0.021],[-1,3]])

    inits_one = np.concatenate((inits_one, inits_two),0)
    inits_two = inits_four
    inits_most = np.concatenate((inits_one,inits_two),0)
    inits = np.row_stack((inits_most,inits_extra))

    # Loop through the inits, solving and plotting
    for i in range(inits.shape[0]):
        # Solve the init
        sol = scp.solve_ivp(f, tspan, inits[i,:], atol = 1e-12, rtol = 1e-12)
        
        # Trim to only get points on the plot
        x_trim_sol = sol.y[:,(sol.y[0,:] <= x_limits[1]) & (sol.y[0,:] >= x_limits[0])]
        sol_s = x_trim_sol[:,(x_trim_sol[1,:] <= y_limits[1]) & (x_trim_sol[1,:] >= y_limits[0])]

        # Define an approximate arc-length for each solution
        arc_length = [0]
        split = 0
        if sol_s.shape[1] > 1:
            for i in range(1,sol_s.shape[1]):
                seg_length = math.sqrt(sol_s[0,i]**2 + sol_s[1,i]**2)
                arc_length += [arc_length[-1] + seg_length]
                if (seg_length >= 0.5):
                    split = len(arc_length)
        
        if split == 0:
            sol_list += [sol_s]
        else:
            sol_s_1 = sol_s[:,:split]
            sol_s_2 = sol_s[:,split:]
            sol_list += [sol_s_1, sol_s_2]
        
        
        
    for sol in sol_list:   
        ax.plot(sol[0,:],sol[1,:], color = 'steelblue')
        
        # Find an approximate arc length of the curve
        arc_length = [0]
        if sol.shape[1] > 1:
            for i in range(1,sol.shape[1]):
                seg_length = math.sqrt((sol[0,i]-sol[0,i-1])**2 + (sol[1,i]-sol[1,i-1])**2)
                arc_length += [arc_length[-1] + seg_length]
                if (seg_length >= 5):
                    split = len(arc_length)
                    break
            mid_point = arc_length[-1]/2
            
            mid = bisect_left(arc_length,mid_point)
            dx = sol[0,mid+1] - sol[0,mid]
            dy = sol[1,mid+1] - sol[1,mid]
            # Plot an arrow on the curve for this arc length
            ax.arrow(sol[0,mid],sol[1,mid],dx,dy,head_width = 0.1, 
                head_length = 0.1, overhang = 0.5, linestyle = None, edgecolor = 'None', 
                facecolor = 'steelblue')

    sol_list_neg = []
    inits_neg = np.array([[-0.03,-0.03],[0.03,0.03]])
    tspan_neg = [0,-10]
    # Now do the same for the solutions being run backwards
    for i in range(inits_neg.shape[0]):
        # Solve the init
        sol = scp.solve_ivp(f, tspan_neg, inits_neg[i,:], atol = 1e-12, rtol = 1e-12)
        
        # Trim to only get points on the plot
        x_trim_sol = sol.y[:,(sol.y[0,:] <= x_limits[1]) & (sol.y[0,:] >= x_limits[0])]
        sol_s = x_trim_sol[:,(x_trim_sol[1,:] <= y_limits[1]) & (x_trim_sol[1,:] >= y_limits[0])]

        # Define an approximate arc-length for each solution
        arc_length = [0]
        split = 0
        if sol_s.shape[1] > 1:
            for i in range(1,sol_s.shape[1]):
                seg_length = math.sqrt(sol_s[0,i]**2 + sol_s[1,i]**2)
                arc_length += [arc_length[-1] + seg_length]
                if (seg_length >= 0.5):
                    split = len(arc_length)
        
        if split == 0:
            sol_list_neg += [sol_s]
        else:
            sol_s_1 = sol_s[:,:split]
            sol_s_2 = sol_s[:,split:]
            sol_list_neg += [sol_s_1, sol_s_2]
        
        
    # Plot all these points
    for sol in sol_list_neg:   
        ax.plot(sol[0,:],sol[1,:], color = 'steelblue')
        
        # Find an approximate arc length of the curve
        arc_length = [0]
        if sol.shape[1] > 1:
            for i in range(1,sol.shape[1]):
                seg_length = math.sqrt((sol[0,i]-sol[0,i-1])**2 + (sol[1,i]-sol[1,i-1])**2)
                arc_length += [arc_length[-1] + seg_length]
                if (seg_length >= 5):
                    split = len(arc_length)
                    break
            mid_point = arc_length[-1]/2
            
            mid = bisect_left(arc_length,mid_point)
            dx = -sol[0,mid+1] + sol[0,mid]
            dy = -sol[1,mid+1] + sol[1,mid]
            # Plot an arrow on the curve for this arc length
            ax.arrow(sol[0,mid],sol[1,mid],dx,dy,head_width = 0.1, 
                head_length = 0.1, overhang = 0.5, linestyle = None, edgecolor = 'None', 
                facecolor = 'steelblue')
    stable_fp = matplotlib.patches.Circle((-2,1), radius = 0.05, color = 'black', zorder = 3)
    unstable_fp = matplotlib.patches.Circle((0,0), radius = 0.05, edgecolor = 'black', facecolor='None', zorder = 3)
    ax.add_patch(stable_fp)
    ax.add_patch(unstable_fp)
    if show == 1:
        plt.show()  
    return(sol_list)

import numpy as np
def f(t,y):
    ydot = np.array([y[0]**2-y[1]+y[0]*y[1]-y[1]**2,-0.5*y[0]-y[1]])
    return(ydot)

phase_sketch(f, [-3, 3], [-3, 3], 5, 1, [-3, 3], [-3, 3])