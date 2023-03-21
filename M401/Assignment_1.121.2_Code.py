# Plot some data
import numpy as np
import matplotlib.pyplot as plt

# create some data
x_vals = np.linspace(0,100,400) # these are the values that go along your x-axis (e.g. days)
y_vals = np.linspace(0,100,400) # these are the values that go along your y-axis (e.g. ash fall per day)

#
fig, ax = plt.subplots(1,1) # can change these numbers to put multiple plots on one figure 
ax.plot(x_vals, y_vals) # we can plot a line
ax.plot(x_vals, 100-y_vals) # and we can put multiple lines on one figure
ax.plot(x_vals, 20*np.sin(x_vals), 'o', color = 'seagreen', label = 'Circles') # we can tinker 
# with line styles and color as well.
# The detail on line styles (and other arguments) is here https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
# and the list of colors is here https://matplotlib.org/stable/gallery/color/named_colors.html

# You can also tweak other quantities
ax.set_xlabel('Days', labelpad = -2, loc = 'right', size = 14)
ax.set_ylabel('Ashfall', labelpad = -10, loc = 'top', rotation = 'horizontal', size = 14)
ax.set_xlim((0,100))
ax.set_ylim((0,150))
ax.legend() # adds a legend with lines labeled according to labels set earlier

# call this so the final plot is displayed. When a figure window is open Python will not run anything 
# else, so make sure to close it when you are finished!
plt.show()