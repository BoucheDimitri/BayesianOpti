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


def plot_acq_func_2d(xmat,
                     y,
                     Rinv,
                     beta_hat,
                     theta,
                     p,
                     bounds,
                     grid_size,
                     xi=0,
                     acq_func_key="EI",
                     **kwargs):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    xgrid, ygrid = mesh_grid(bounds, grid_size)
    zgrid = np.zeros(shape=xgrid.shape)
    for i in range(0, xgrid.shape[0]):
        for j in range(0, xgrid.shape[1]):
            zgrid[i, j] = am.acq_func(xmat,
                                      np.array([xgrid[i, j], ygrid[i, j]]),
                                      y,
                                      Rinv,
                                      beta_hat,
                                      theta,
                                      p,
                                      xi=xi,
                                      acq_func_key=acq_func_key,
                                      **kwargs)
    surf = ax.plot_surface(xgrid, ygrid, zgrid, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.show()


def plot_test_func_2d(bounds,
                      grid_size,
                      test_func_key="Mystery"):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    xgrid, ygrid = mesh_grid(bounds, grid_size)
    zgrid = np.zeros(shape=xgrid.shape)
    for i in range(0, xgrid.shape[0]):
        for j in range(0, xgrid.shape[1]):
            zgrid[i, j] = tf.funcs_dic[test_func_key](np.array([xgrid[i, j], ygrid[i, j]]))
    surf = ax.plot_surface(xgrid, ygrid, zgrid, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.show()


