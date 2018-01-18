import numpy as np
import math


def kernel_func(x1, x2, theta, p):
    pdist = np.power(abs(x1 - x2), p)
    return np.exp(-theta * pdist)


def kernel_func_classic(xvec1, xvec2, theta, p):
	"""
	Classic kernel func using euclidian distance and not product of kernels
	"""
	pdist = np.power(np.absolute(xvec1 - xvec2), p)
	return np.exp(-(1.0 / theta) * pdist)


def kernel_mat_classic(xmat, theta, p):
    n = xmat.shape[0]
    R = np.zeros((n, n))
    # We use the symmetric structure to divide
    # the number of calls to corr_func_2d by 2
    for j in range(0, n):
        for i in range(j, n):
            corr = kernel_func_classic(xmat[i, :], xmat[j, :], theta, p)
            R[i, j] = corr
            R[j, i] = corr
    return R


def kernel_rx_classic(xmat, xnew, theta, p):
    """
    Compute rx correlation vector for new data point using classic kernel approach

    Args:
        xmat (numpy.ndarray) : the data points, shape = (n, k),
        xnew (numpy.ndarray) : the new data vec, shape = (k, )
        theta (float) :  theta parameter in kernel_function_classic
        p (float) :  p parameter in kernel_function_classic
    """
    n = xmat.shape[0]
    rx = np.zeros((n, 1))
    for i in range(0, n):
        rx[i, 0] = kernel_func(xmat[i, :], xnew, theta, p)
    return rx



def kernel_func_2d_prod(xvec1, xvec2, theta_vec, p_vec):
    """
    2d version of correlation kernel using product rule

    Args:
            xvec1 (numpy.ndarray) : first datapoint, shape = (2, )
            xvec2 (numpy.ndarray) : second datapoint, shape = (2, )
            theta_vec (numpy.ndarray) : vector of theta params, one by dim, shape = (2, )
            p_vec (numpy.ndarray) : powers used to compute the distance, one by dim, shape = (2, )

    Returns:
            float. The 2d kernel correlation between xvec1 and xvec2
    """
    firstdim = kernel_func(xvec1[0], xvec2[0], theta_vec[0], p_vec[0])
    secdim = kernel_func(xvec1[1], xvec2[1], theta_vec[1], p_vec[1])
    return firstdim * secdim


def kernel_mat_2d_prod(xmat, theta_vec, p_vec):
    """
    Compute correlation matrix for a set of 2d points

    Args:
        xmat (numpy.ndarray) : shape = (n, 2)
        theta_vec (numpy.ndarray) : vector of theta params, one by dim, shape = (2, )
        p_vec (numpy.ndarray) : powers used to compute the distance, one by dim, shape = (2, )

    Returns:
        numpy.ndarray. The correlation matrix
    """
    n = xmat.shape[0]
    R = np.zeros((n, n))
    # We use the symmetric structure to divide
    # the number of calls to corr_func_2d by 2
    for j in range(0, n):
        for i in range(j, n):
            corr = kernel_func_2d_prod(xmat[i, :], xmat[j, :], theta_vec, p_vec)
            R[i, j] = corr
            R[j, i] = corr
    return R


def kernel_rx_2d_prod(xmat, xnew, theta_vec, p_vec):
    """
    Compute rx correlation vector for new data point

    Args:
            xmat (numpy.ndarray) : the data points, shape = (n, 2),
            xnew (numpy.ndarray) : the new data vec, shape = (2, )
            theta_vec (numpy.ndarray) : vector of theta params, one by dim, shape = (2, )
    p_vec (numpy.ndarray) : powers used to compute the distance, one by dim, shape = (2, )
    """
    n = xmat.shape[0]
    rx = np.zeros((n, 1))
    for i in range(0, n):
        rx[i, 0] = kernel_func_2d_prod(xmat[i, :], xnew, theta_vec, p_vec)
    return rx
