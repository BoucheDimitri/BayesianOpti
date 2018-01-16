import math
import numpy as np
import scipy.linalg as linalg
import scipy.optimize as optimize

import prediction_formulae as pred 
import cho_inv
import gp_tools


def params_to_vec(params_vec):
	"""
	Separate parameters stacked together (params_vec = [theta_1, theta_2, p])
	We stack them together to be able to use scipy optimization functions

	Args :
        params_vec (numpy.ndarray) : shape = (3, ), [theta_1, theta_2, p]
	
	Returns :
		tuple. (theta_vec, p)
	"""
	theta_vec = params_vec[0:2]
	p = float(params_vec[2])
	return theta_vec, p


def hat_sigmaz_sqr_mle(y, R, params_vec):
	"""
	Since R depends on theta and p, so does Rinv and beta estimate
	We need to have hat_sigmaz as a function of theta and p for MLE

	Args :
        y (numpy.ndarray) : shape = (n, 1)
        R (numpy.ndarray) : Kernel matrix
        params_vec (numpy.ndarray) : shape = (3, ), [theta_1, theta_2, p]

    Returns :
    	float. estimation of sigmaz_sqr
	"""
	theta_vec, p = params_to_vec(params_vec)
	#Rinv = cho_inv.cholesky_inv(R)
	Rinv = np.linalg.inv(R)
	hat_beta = pred.beta_est_bis(y, Rinv)
	return pred.hat_sigmaz_sqr(y, Rinv, hat_beta)


def log_likelihood(xmat, y, params_vec):
	"""
	Log likelihood, params_vec = [theta_1, theta_2, p]

	Args :
		xmat (numpy.ndarray) : shape = (n, 2)
        y (numpy.ndarray) : shape = (n, 1)
        params_vec (numpy.ndarray) : shape = (3, ), [theta_1, theta_2, p]

    Returns :
    	float. log likelihood
	"""
	theta_vec, p = params_to_vec(params_vec)
	R = gp_tools.kernel_mat(xmat, theta_vec, p)
	n = R.shape[0]
	Rinv = np.linalg.inv(R)
	detR = np.linalg.det(R)
	hat_sigz_sqr = hat_sigmaz_sqr_mle(y, R, params_vec)
	print("sigma " + str(hat_sigz_sqr))
	print("Det " + str(detR))
	return - 0.5 * (n * math.log(hat_sigz_sqr) + math.log(detR))


def max_log_likelihood(xmat, y, xinit):
	#We have an issue with the determinant which gets very 
	#very small to the point that a math domain error is raised
	minus_llk_opti = lambda params : - log_likelihood(xmat, y, params)
	opt = optimize.minimize(fun=minus_llk_opti, x0=xinit, method="L-BFGS-B")
	return opti






