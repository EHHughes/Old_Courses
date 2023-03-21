def phase_sketch(f, inits, inits_neg, ax_obj, stable, half_stable, unstable, limits):
    # Import the packages we need and set up the plot to use latex
    #import matplotlib.pyplot as plt
    import matplotlib
    import scipy.integrate as scp
    import math
    from bisect import bisect_left
    plt.sca(ax_obj)
    ax_obj.set_xlim(limits[0],limits[1])
    ax_obj.set_ylim(limits[0],limits[1])
    ax_obj.set_xlabel('$x$', loc = 'right', fontsize = 18)
    ax_obj.set_ylabel('$y$', loc = 'top', rotation = 'horizontal', fontsize = 18)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.tight_layout()

    # Set a tspan and unpack x and y limits into linspace
    tspan = [0,10]
    tspan_neg = [0,-10]

    # Create a list to hold all our solutions
    sol_list = []
    sol_list_neg = []

    # Loop through the inits, solving
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
                seg_length = math.sqrt((sol_s[0,i]-sol_s[0,i-1])**2 + (sol_s[1,i]-sol_s[1,i-1])**2)
                arc_length += [arc_length[-1] + seg_length]
                # If the arc-length jumps suddenly, introduce a split here so they can 
                # be plotted separately
                if (seg_length >= 0.5):
                    split = len(arc_length)
        
        # If there is no split, add to the solution directly
        if split == 0:
            sol_list += [sol_s]
        else:
            # Split the solution up and add both (assumes at most one split)
            sol_s_1 = sol_s[:,:split]
            sol_s_2 = sol_s[:,split:]
            sol_list += [sol_s_1, sol_s_2]
        
        
    # Plot all these points
    for sol in sol_list:   
        ax_obj.plot(sol[0,:],sol[1,:], color = 'steelblue')
        
        # Redo the arc length for these split solutions
        arc_length = [0]
        if sol.shape[1] > 5:
            for i in range(1,sol.shape[1]):
                seg_length = math.sqrt((sol[0,i]-sol[0,i-1])**2 + (sol[1,i]-sol[1,i-1])**2)
                arc_length += [arc_length[-1] + seg_length]
            mid_point = arc_length[-1]/2
            
            # Find the mid point and arrow here
            mid = bisect_left(arc_length,mid_point)
            dx = sol[0,mid+1] - sol[0,mid]
            dy = sol[1,mid+1] - sol[1,mid]
            # Plot an arrow on the curve for this arc length
            ax_obj.arrow(sol[0,mid],sol[1,mid],dx,dy,head_width = 0.2, 
                head_length = 0.1, overhang = 0.5, linestyle = None, edgecolor = 'None', 
                facecolor = 'steelblue')

    # Now do the same for the solutions being run backwards
    if inits_neg is not None:
        for i in range(inits_neg.shape[0]):
            # Solve the init
            sol = scp.solve_ivp(f, tspan_neg, inits_neg[i,:], atol = 1e-10, rtol = 1e-12)
            
            # Trim to only get points on the plot
            x_trim_sol = sol.y[:,(sol.y[0,:] <= x_limits[1]) & (sol.y[0,:] >= x_limits[0])]
            sol_s = x_trim_sol[:,(x_trim_sol[1,:] <= y_limits[1]) & (x_trim_sol[1,:] >= y_limits[0])]

            # Define an approximate arc-length for each solution
            arc_length = [0]
            split = 0
            if sol_s.shape[1] > 5:
                for i in range(1,sol_s.shape[1]):
                    seg_length = math.sqrt((sol_s[0,i]-sol_s[0,i-1])**2 + (sol_s[1,i]-sol_s[1,i-1])**2)
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
            ax_obj.plot(sol[0,:],sol[1,:], color = 'steelblue')
            
            # Find an approximate arc length of the curve
            arc_length = [0]
            if sol.shape[1] > 1:
                for i in range(1,sol.shape[1]):
                    seg_length = math.sqrt((sol[0,i]-sol[0,i-1])**2 + (sol[1,i]-sol[1,i-1])**2)
                    arc_length += [arc_length[-1] + seg_length]
                mid_point = arc_length[-1]/2
                
                mid = bisect_left(arc_length,mid_point)
                dx = -sol[0,mid+1] + sol[0,mid]
                dy = -sol[1,mid+1] + sol[1,mid]
                # Plot an arrow on the curve for this arc length
                ax_obj.arrow(sol[0,mid],sol[1,mid],dx,dy,head_width = 0.2, 
                    head_length = 0.1, overhang = 0.5, linestyle = None, edgecolor = 'None', 
                    facecolor = 'steelblue')

    # Add the half-stable fixed points if there are any
    if half_stable is not None:
        for pt in half_stable:
            w1 = matplotlib.patches.Wedge(pt, 0.05, -90, 90, edgecolor = 'black', facecolor = 'None', zorder = 3)
            w2 = matplotlib.patches.Wedge(pt, 0.05, 90, -90, color = 'black', zorder = 3)
            ax_obj.add_patch(w1)
            ax_obj.add_patch(w2) 
    if stable is not None:
        for pt in stable:
            stable_fp = matplotlib.patches.Circle(pt, radius = 0.05, color = 'black', zorder = 3)
            ax_obj.add_patch(stable_fp)
    if unstable is not None:
        for pt in unstable:
            stable_fp = matplotlib.patches.Circle(pt, radius = 0.05, edgecolor = 'black', facecolor = 'None', zorder = 3)
            ax_obj.add_patch(stable_fp)
    return(sol_list)

def f1(t,y):
    ydot = np.array([y[0]**2-y[1]+y[0]*y[1]-y[1]**2,0.5*y[0]-y[1]])
    return(ydot)


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern"]})
tspan = [0,30]
tspan_neg = [0,-10]
x_limits = (-3,3)
y_limits = (-3,3)

fig, (ax1, ax2, ax3) = plt.subplots(1,3)


# Create the first set of initial conditions
num_curves = 5
x_inits = np.linspace(x_limits[0],x_limits[1],num=num_curves)
y_inits = np.linspace(y_limits[0],y_limits[1],num=num_curves)

inits_one = np.column_stack((x_inits,y_inits[0]*np.ones((num_curves,1))))
inits_two = np.column_stack((x_inits[0]*np.ones((num_curves,1)),y_inits))
inits_three = np.column_stack((x_inits, y_limits[-1]*np.ones((num_curves,1))))
inits_four = np.column_stack((x_limits[-1]*np.ones((num_curves,1)),y_inits))
inits_extra = np.array([[2.2,-3],[2.8,-3],[2,3],[2.5,3],[2.1,3],[2.15,3],
    [0.35,0.2],[0.44,0.22]])

inits_one = np.concatenate((inits_one, inits_two),0)
inits_two = np.concatenate((inits_three, inits_four),0)
inits_most = np.concatenate((inits_one,inits_two),0)
inits_one = np.row_stack((inits_most,inits_extra))

neg_inits_one = np.array([[3,1.4],[3,0.8],[0.4,0.24],[0.38,0.13]])

# Plot the first set mu > 0
phase_sketch(f1, inits_one, neg_inits_one, ax3, [(0,0)], None,[(2/5,1/5)], x_limits)


# Generate the second set of initial conditions mu = 0
def f2(t,y):
    ydot = np.array([y[0]**2-y[1]+y[0]*y[1]-y[1]**2,-y[1]])
    return(ydot)

x_inits = np.linspace(x_limits[0],x_limits[1],num=num_curves)
y_inits = np.linspace(y_limits[0],y_limits[1],num=num_curves)

# Set up initial conditions (configure as appropriate)
inits_one = np.column_stack((x_inits,y_inits[0]*np.ones((num_curves,1))))
inits_two = np.column_stack((x_inits[0]*np.ones((num_curves,1)),y_inits))
inits_four = np.column_stack((x_limits[-1]*np.ones((num_curves,1)),y_inits))
inits_extra = np.array([[1.5,3],[2.5,3],[3,-2.5],[-1,3], [2,3], [2.3,3]])

inits_one = np.concatenate((inits_one, inits_two),0)
inits_two = inits_four
inits_most = np.concatenate((inits_one,inits_two),0)
inits_two = np.row_stack((inits_most,inits_extra))

inits_neg_two = np.array([[3,0],[3,1]])

phase_sketch(f2,inits_two,inits_neg_two,ax2, None,[(0,0)],None, x_limits)

# Plot the first set of initial conditions mu < 0
def f3(t,y):
    ydot = np.array([y[0]**2-y[1]+y[0]*y[1]-y[1]**2,-0.5*y[0]-y[1]])
    return(ydot)

inits_one = np.column_stack((x_inits,y_inits[0]*np.ones((num_curves,1))))
inits_two = np.column_stack((x_inits[0]*np.ones((num_curves,1)),y_inits))
inits_four = np.column_stack((x_limits[-1]*np.ones((num_curves,1)),y_inits))
inits_extra = np.array([[1.5,3],[2,3],[3,-2],[1.9,3],[1.95,3],[1.93,3],[3,-2.5],
    [0.031,-0.021],[-0.031,0.021],[-1,3]])

inits_one = np.concatenate((inits_one, inits_two),0)
inits_two = inits_four
inits_most = np.concatenate((inits_one,inits_two),0)
inits_three = np.row_stack((inits_most,inits_extra))

inits_neg_three = np.array([[-0.03,-0.03],[0.03,0.03]])

phase_sketch(f3,inits_three,inits_neg_three,ax1,[(-2,1)],None,[(0,0)], x_limits)

plt.show()