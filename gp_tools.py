import numpy as np
from itertools import product
import math


def corr_func(x1, x2, theta, p):
	pdist = math.pow(abs(x1 - x2), p)
	return math.exp(-theta * pdist)


def corr_func_2d(xvec1, xvec2, theta_vec, p):
	firstdim = corr_func(xvec1[0], xvec2[0], theta_vec[0], p)
	secdim = corr_func(xvec1[1], xvec2[1], theta_vec[1], p)
	return firstdim * secdim


def corr_mat(xmat, theta_vec, p):
	"""

	Args:
	    xmat (numpy.ndarray) : must be of shape = (n, 2)
	Returns:
	    numpy.ndarray. The correlation matrix
	"""
	n = xmat.shape[0]
	R = np.zeros((n, n))
	#We use the symmetric structure to divide
	#the number of calls to corr_func_2d by 2
	for j in range(0, n):
		for i in range(j, n):
			corr = corr_func_2d(xmat[i, :], xmat[j, :], theta_vec, p)
			R[i, j] = corr
			R[j, i] = corr
	return R



#xtest = np.random.rand(100, 2)
#theta_vec = [5, 5]
#p = 1.5
#R = corr_mat(xtest, theta_vec, p)
#print(R)

    