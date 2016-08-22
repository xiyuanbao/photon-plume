# This program creates a BOS pattern consisting of black or white dots

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

def circles(x, y, s, c='b', vmin=None, vmax=None, **kwargs):
    """
    Make a scatter of circles plot of x vs y, where x and y are sequence
    like objects of the same lengths. The size of circles are in data scale.

    Parameters
    ----------
    x,y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, )
        Radius of circle in data unit.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    a = np.arange(11)
    circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """


    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None
    if 'fc' in kwargs: kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs: kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs: kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs: kwargs.setdefault('linewidth', kwargs.pop('lw'))

    patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    if c is not None:
        plt.sci(collection)
    return collection



# these are the minimum and maximum X,Y co-ordinates (microns) in the object space for the points on the pattern
X_min = -7.5e4
X_max = 7.5e4
Y_min = -7.5e4
Y_max = 7.5e4

# this is the Z co-ordinate of the pattern (in microns)
Z = 0.0

# this is the diameter of a dot in the bos pattern in microns
dot_diameter = 3.0e2

# this is the spacing for the grid over which the dots will be randomly distributed
grid_spacing = 2.0 * dot_diameter

# these are the number of grid points over X and Y
num_grid_points = (X_max - X_min)/grid_spacing

# this is the ratio of the number of dots to the number of grid points. this ratio should be less than 1
density_dots = 0.9

# this is the number of dots to be generated along each dimension
num_dots = density_dots * num_grid_points

# this displays the parameters of the texture to the user
print "number of grid points:", num_grid_points
print "number of dots:", num_dots

# this randomly generates integers corresponding to the X and Y indices in the grid
xy_loc_int = np.random.randint(0, high=num_grid_points, size=(num_dots,2))

# this converts integer locations to positions in real space
xy_loc_float = np.zeros(xy_loc_int.shape, dtype=np.float32)
xy_loc_float[:,0] = X_min + xy_loc_int[:,0]*grid_spacing
xy_loc_float[:,1] = X_min + xy_loc_int[:,1]*grid_spacing

# this creates a figure where the circles will be drawn
fig = plt.figure(1, figsize=(8,8))
fig.set_size_inches(8,8)
ax = fig.add_subplot(111, axisbg='black')

# this draws circles whose centers are the co-ordinates in xy_loc_float and the radius is the dot_diameter
# out = circles(xy_loc_float[:,0], xy_loc_float[:,1], 2*dot_diameter*np.ones((num_dots,1)), c= 'w', alpha=0.5, ec='none')
plt.scatter(xy_loc_float[:,0], xy_loc_float[:,1], s=9, c='w', edgecolors='none')
ax.set_xlim([X_min, X_max])
ax.set_ylim([Y_min, Y_max])
# ax.set_autoscale_on(False)
ax.set_xticks([])
ax.set_yticks([])

plt.axis('equal')
# plt.show()

fig.savefig('bos_scatter_test_circles.png', bbox_inches='tight', pad_inches=0)

