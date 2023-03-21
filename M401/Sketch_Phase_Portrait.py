def phase_sketch(f, x_limits, y_limits, num_curves, show, plt_xlim, plt_ylim):
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.integrate as scp
    import math
    from bisect import bisect_left
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern"]})

    tspan = [0,20]
    tspan_neg = [0,-2]
    x_inits = np.linspace(x_limits[0],x_limits[1],num=num_curves)
    y_inits = np.linspace(y_limits[0],y_limits[1],num=num_curves)

    inits_one = np.column_stack((x_inits,y_inits[0]*np.ones((num_curves,1))))
    inits_two = np.column_stack((x_inits[0]*np.ones((num_curves,1)),y_inits))
    inits_three = np.column_stack((x_inits, y_limits[-1]*np.ones((num_curves,1))))
    inits_four = np.column_stack((x_limits[-1]*np.ones((num_curves,1)),y_inits))
    inits_extra = np.array([[0.8,1],[0.80,-1],[0.9,1]])

    inits_one = np.concatenate((inits_one, inits_two),0)
    inits_two = np.concatenate((inits_three, inits_four),0)
    inits_most = np.concatenate((inits_one,inits_two),0)
    inits = np.row_stack((inits_most,inits_extra))

    sol_list_pos = []
    #sol_list_neg = []

    for i in range(num_curves*4+3):
        sol_list_pos += [scp.solve_ivp(f, tspan, inits[i,:], atol = 1e-12, rtol = 1e-12)]
    #    sol_list_neg += [scp.solve_ivp(f, tspan_neg, inits[i,:], atol = 1e-12, rtol = 1e-12)]

    sol_list = sol_list_pos
    if show == 1:
        fig = plt.figure()
        ax = fig.add_subplot()
        for sol in sol_list_pos:
            x_trim_sol = sol.y[:,(sol.y[0,:] <= x_limits[1]) & (sol.y[0,:] >= x_limits[0])]
            sol_s = x_trim_sol[:,(x_trim_sol[1,:] <= y_limits[1]) & (x_trim_sol[1,:] >= y_limits[0])] 
            ax.plot(sol_s[0,:],sol_s[1,:], color = 'steelblue')
            arc_length = [0]
            if sol_s.shape[1] > 1:
                for i in range(1,sol_s.shape[1]):
                    arc_length += [arc_length[i-1] + math.sqrt(sol_s[0,i]**2 + sol_s[1,i]**2)]
                mid_point = arc_length[-1]/2
                mid = bisect_left(arc_length,mid_point)
                dx = sol_s[0,mid+1] - sol_s[0,mid]
                dy = sol_s[1,mid+1] - sol_s[1,mid]
                ax.arrow(sol_s[0,mid],sol_s[1,mid],dx,dy,head_width = 0.05, head_length = 0.05, overhang = 0.5, linestyle = None, edgecolor = 'None', facecolor = 'steelblue')
        #for sol_neg in sol_list_neg:
        #    plt.plot(sol_neg.y[0,:],sol_neg.y[1,:], color = 'steelblue')
        plt.xlim(plt_xlim)
        plt.ylim(plt_ylim)
        ax.set_xlabel('$y$', loc = 'right', fontsize = 18)
        ax.set_ylabel('$x$', loc = 'top', rotation = 'horizontal', fontsize = 18)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.tight_layout()
        plt.show()
    return(sol_list)

import numpy as np
def f(t,y):
    ydot = np.array([y[0]**2-y[1]+y[0]*y[1]-y[1]**2,0.5*y[0]-y[1]])
    return(ydot)

phase_sketch(f, [-1, 1], [-1, 1], 5, 1, [-1,1], [-1,1])