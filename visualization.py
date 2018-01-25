import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import acquisition_max as am


def plot_func_1d(bounds, grid_size, func, nsub_plots=1, axis=None, label=None, c=None):
    if isinstance(bounds[0], tuple):
        grid = np.linspace(bounds[0][0], bounds[0][1], grid_size)
    else:
        grid = np.linspace(bounds[0], bounds[1], grid_size)
    y = [func(np.array([grid[i]])) for i in range(0, grid.shape[0])]
    if axis:
        axis.plot(grid, y, label=label, c=c)
    else:
        fig, axes = plt.subplots(nsub_plots, 1, sharex=True)
        axes[0].plot(grid, y, label=label, c=c)
        return axes


def plot_acq_func_1d(xmat,
                     y,
                     Rinv,
                     beta_hat,
                     theta,
                     p,
                     bounds,
                     grid_size,
                     acq_func,
                     axis):
    """
    3d heated colored surface plot of acquisition function

    Args :
        xmat (numpy.ndarray) : the data points so far, shape = (n, k)
        y (numpy.ndarray) : y, shape=(n, 1)
        Rinv (numpy.ndarray) : inverse of kernel matrix
        beta_hat (float) : estimation of beta
        theta (numpy.ndarray) : vector of theta params, one by dim, shape = (k, )
        p (numpy.ndarray) : powers used to compute the distance, one by dim, shape = (k, )
        bounds (tuple) : ((min_d1, min_d2), (max_d1, max_d2))
        grid_size (tuple) : (gridsize_x, gridsize_y)
        acq_func : Instance of one of the classes in Acquisition_Functions.py file

    Returs:
        nonetype. None

    """
    def acq_plot(xnew):
        return am.complete_acq_func(xmat,
                                    xnew,
                                    y,
                                    Rinv,
                                    beta_hat,
                                    theta,
                                    p,
                                    acq_func)
    axis = plot_func_1d(bounds, grid_size, acq_plot, axis=axis)
    return axis


def add_points_1d(ax, points_x, points_y, c=None, label=None):
    if label:
        ax.scatter(points_x, points_y, c=c, label=label)
    else:
        ax.scatter(points_x, points_y, c=c)
    return ax


def mesh_grid(bounds, grid_size):
    """
    Create a meshgrid for 3d plots

    Args :
        bounds (tuple) : ((min_d1, min_d2), (max_d1, max_d2))
        grid_size (tuple) : (gridsize_x, gridsize_y)

    Returns :
        tuple. Tuple of numpy array
    """
    x_axis = np.linspace(bounds[0][0], bounds[0][1], grid_size[0])
    y_axis = np.linspace(bounds[1][0], bounds[1][1], grid_size[1])
    xgrid, ygrid = np.meshgrid(x_axis, y_axis)
    return xgrid, ygrid


def plot_func_2d(bounds, grid_size, func, title=None):
    """
    3d heated colored surface plot of R^2 to R function

    Args :
        bounds (tuple) : ((min_d1, min_d2), (max_d1, max_d2))
        grid_size (tuple) : (gridsize_x, gridsize_y)
        func (function) : the function to plot
        title (str) : the title for the plot

    Returs:
        nonetype. None

    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    xgrid, ygrid = mesh_grid(bounds, grid_size)
    zgrid = np.zeros(shape=xgrid.shape)
    for i in range(0, xgrid.shape[0]):
        for j in range(0, xgrid.shape[1]):
            zgrid[i, j] = func(np.array([xgrid[i, j], ygrid[i, j]]))
    surf = ax.plot_surface(xgrid, ygrid, zgrid, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.title(title)
    plt.show()


def plot_acq_func_2d(xmat,
                     y,
                     Rinv,
                     beta_hat,
                     theta,
                     p,
                     bounds,
                     grid_size,
                     acq_func):
    """
    3d heated colored surface plot of acquisition function

    Args :
        xmat (numpy.ndarray) : the data points so far, shape = (n, k)
        y (numpy.ndarray) : y, shape=(n, 1)
        Rinv (numpy.ndarray) : inverse of kernel matrix
        beta_hat (float) : estimation of beta
        theta (numpy.ndarray) : vector of theta params, one by dim, shape = (k, )
        p (numpy.ndarray) : powers used to compute the distance, one by dim, shape = (k, )
        bounds (tuple) : ((min_d1, min_d2), (max_d1, max_d2))
        grid_size (tuple) : (gridsize_x, gridsize_y)
        acq_func : Instance of one of the classes in Acquisition_Functions.py file

    Returs:
        nonetype. None

    """
    def acq_plot(xnew):
        return am.complete_acq_func(xmat,
                                    xnew,
                                    y,
                                    Rinv,
                                    beta_hat,
                                    theta,
                                    p,
                                    acq_func)
    plot_func_2d(bounds, grid_size, acq_plot, title=acq_func.name)
