import numpy as np
import scipy.stats as stats
from matplotlib import colors, cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Classes to process cursor movements and distribution
class Cursor(object):
    def __init__(self, ax, dist):

        # Axes object
        self.ax = ax
        self.dist = dist
        self.mappable = cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap='bwr')
        self.color_b = None

        # Highest value of y for normalizing the text location
        self.ylim = ax.get_ylim()[1]

        # previously chosen value of y
        self.y1 = None
        self.y2 = None

        # Defining holders for horizontal lines
        self.lx1 = ax.axhline(color='k', lw=0.5)  # the horiz line
        self.lx2 = ax.axhline(color='k', lw=0.5)  # the horiz line 2
        self.lx3 = ax.axhline(color='k', lw=0.5)  # the horiz line 3

        # Whether lx1 or lx2 have been fixed
        self.fix_lx1 = False
        self.fix_lx2 = False

        # Defining holders for the text values for each y value
        self.txt1 = ax.text(0.5, 0.5, '', transform=ax.transAxes, ma='left')
        self.txt2 = ax.text(0.5, 0.5, '', transform=ax.transAxes, ma='left')
        self.txt3 = ax.text(0.5, 0.5, '', transform=ax.transAxes, ma='left')

        # Holder to define the fill between the successive values of Ys
        self.fill = None

    def mouse_move(self, event):
        # If the mouse is not in the axes, do not do anything
        if not event.inaxes:
            return

        # Get the current Y position of the cursor
        y = event.ydata

        # If the first line is not fixed yet, set its current height to y
        if not self.fix_lx1:
            self.set_y(self.lx1, self.txt1, y)

        # If the first line is fixed yet, but the second is not set its current height to y
        if self.fix_lx1 and not self.fix_lx2:

            # If the fill area already exists, remove it to make space for the new one
            if self.fill is not None:
                self.fill.remove()
            self.y2 = y

            # after deleting the previous fill area, a new one is added
            self.fill = self.ax.fill_between(self.ax.get_xlim(), self.y1, self.y2, color='#C3C3C3', alpha=0.3)
            self.set_y(self.lx2, self.txt2, y)

        # If both lines have been fixed, set the third one as an auxillary line to let the user select another value
        if self.fix_lx1 and self.fix_lx2:
            self.set_y(self.lx3, self.txt3, y)

    def mouse_click(self, event):
        # If the mouse is not in the axes, do not do anything
        if not event.inaxes:
            return

        # if both the first and the second lines were fixed prior to the click, free both lines by setting fix_ to False
        if self.fix_lx1 and self.fix_lx2:
            self.fix_lx1 = False
            self.fix_lx2 = False
            self.fill.remove()
            self.fill = self.ax.fill_between(self.ax.get_xlim(), 0, 0, color='#C3C3C3', alpha=0.3)

        # Set the first line to the current y and fix it.
        y = event.ydata
        self.y1 = y
        self.fix_lx1 = True
        self.set_y(self.lx1, self.txt1, y)

    def mouse_release(self, event):
        # If the mouse is not in the axes, do not do anything
        if not event.inaxes:
            return

        # Once the mouse is released, fix line 2, recolor the bars based on the selected values of y and then redraw
        y = event.ydata
        self.fix_lx2 = True
        self.recolor(self.y1, self.y2)
        self.set_y(self.lx2, self.txt2, y)

    def set_y(self, lx, txt, y):

        # This function draws the canvas with the horizontal lines, based on the the selected y values
        # The function also adds the location of the text right above the line
        lx.set_ydata(y)
        txt.set_position((-0.12, (y - 1000) / self.ylim))
        txt.set_text('%1.0f' % (y))
        self.ax.figure.canvas.draw()

    def recolor(self, y1, y2):
        # This function recolors various bars
        # Calculate probability using distribution object methods
        probs = self.dist.range_p(y1, y2)

        # Ensure Rectangular Patch Objects are in the same order as the calculated probability
        ind = np.array([patch.xy[0] for patch in self.ax.patches]).argsort()

        # Scalar Mappable object is used to convert probability values to RGB
        colors_val = self.mappable.to_rgba(probs[0])

        # For each converted value of RGB, the appropriate color is set
        for i in range(probs.shape[1]):
            self.ax.patches[ind[i]].set_color(colors_val[ind[i]])

        # This if condition ensures the color bar is applied only once.
        if self.color_b is None:
            # A separate inset axes is created specifically for the colorbar
            axins = inset_axes(self.ax,
                               width="30%",  # width = 5% of parent_bbox width
                               height="4%",  # height : 50%
                               loc='lower left',
                               bbox_to_anchor=(0.7, 1, 1, 1),
                               bbox_transform=self.ax.transAxes,
                               borderpad=0,
                               )
            # Colorbar is added using the inset axes created above
            self.color_b = self.ax.get_figure().colorbar(self.mappable, cax=axins, orientation="horizontal",
                                                         ticks=[0, 1])
            axins.set_title('Confidence Interval', fontsize='small')


class Distribution(object):
    def __init__(self, means, stds):
        self.means = means
        self.stds = stds
        self.dist = stats.norm(loc=means, scale=stds)

    # Calculates cumulative distribution function at x i.e what is the probability the mean is going to be less than x
    def cdf(self, x):
        return self.dist.cdf(x)

    # Using cdf method, this method calculates the probability the mean of a certain distribution is going to lie in a range
    def range_p(self, x1, x2):
        p = self.cdf(x1) - self.cdf(x2)
        return abs(p)

    # returns all the means
    def get_means(self):
        return self.means

    # returns all the standard deviations
    def get_stds(self):
        return self.stds