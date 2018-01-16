import numpy as np
import math


def kernel_func(x1, x2, theta, p):
    pdist = math.pow(abs(x1 - x2), p)
    return math.exp(-theta * pdist)


def kernel_func_2d(xvec1, xvec2, theta_vec, p):
    """
    2d version of correlation kernel using product rule

    Args:
            xvec1 (numpy.ndarray) : first datapoint, shape = (2, )
            xvec2 (numpy.ndarray) : second datapoint, shape = (2, )
            theta_vec (numpy.ndarray) : vector of theta params, shape = (2, )
            p (float) : power used to compute the distance between xvec1 and xvec2

    Returns:
            float. The 2d kernel correlation between xvec1 and xvec2
    """
    firstdim = kernel_func(xvec1[0], xvec2[0], theta_vec[0], p)
    secdim = kernel_func(xvec1[1], xvec2[1], theta_vec[1], p)
    return firstdim * secdim


def kernel_mat(xmat, theta_vec, p):
    """
    Compute correlation matrix for a set of 2d points

    Args:
        xmat (numpy.ndarray) : shape = (n, 2)

    Returns:
        numpy.ndarray. The correlation matrix
    """
    n = xmat.shape[0]
    R = np.zeros((n, n))
    # We use the symmetric structure to divide
    # the number of calls to corr_func_2d by 2
    for j in range(0, n):
        for i in range(j, n):
            corr = kernel_func_2d(xmat[i, :], xmat[j, :], theta_vec, p)
            R[i, j] = corr
            R[j, i] = corr
    return R


def kernel_rx(xmat, xnew, theta_vec, p):
	"""
	Compute rx correlation vector for new data point

	Args:
		xmat (numpy.ndarray) : the data points, shape = (n, 2), 
		xnew (numpy.ndarray) : the new data vec, shape = (2, )
	"""
	n = xmat.shape[0]
	rx = np.zeros((n, 1))
	for i in range(0, n):
		rx[i, 0] = kernel_func_2d(xmat[i, :], xnew, theta_vec, p)
	return rx

