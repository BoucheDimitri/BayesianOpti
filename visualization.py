import acquisition_functions as af
import acquisition_max as am
import test_functions as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def mesh_grid(bounds, gridsize):
    x_axis = np.linspace(bounds[0][0], bounds[0][1], gridsize[0])
    y_axis = np.linspace(bounds[1][0], bounds[1][1], gridsize[1])
    xgrid, ygrid = np.meshgrid(x_axis, y_axis)
    return xgrid, ygrid


def plot_func_2d(bounds, grid_size, func, title=None):
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
