import math
import numpy as np
import scipy.linalg as linalg
import scipy.optimize as optimize

import prediction_formulae as pred
import cho_inv
import gp_tools


def params_to_vec(params_vec):
    """
    Separate parameters stacked together (params_vec = [theta_1, theta_2, p_1, p_2])
    We stack them together to be able to use scipy optimization functions

    Args :
    	params_vec (numpy.ndarray) : shape = (4, ), [theta_1, theta_2, p_1, p_2]

    Returns :
         tuple. (theta_vec, p_vec)
    """
    theta_vec = params_vec[0:2]
    p_vec = params_vec[2:]
    return theta_vec, p_vec


def hat_sigmaz_sqr_mle(y, R):
    """
	hat_sigmaz depending only on R and and doing the computations
	for beta. Will be useful for Mle estimation

    Args :
    	y (numpy.ndarray) : shape = (n, 1)
    	R (numpy.ndarray) : Kernel matrix

	Returns :
    	float. estimation of sigmaz_sqr
    """
    #Rinv = cho_inv.cholesky_inv(R)
    Rinv = np.linalg.inv(R)
    hat_beta = pred.beta_est_bis(y, Rinv)
    return pred.hat_sigmaz_sqr(y, Rinv, hat_beta)


def log_likelihood(xmat, y, params_vec):
    """
    Log likelihood, params_vec = [theta_1, theta_2, p_1, p_2]

    Args :
        xmat (numpy.ndarray) : shape = (n, 2)
    	y (numpy.ndarray) : shape = (n, 1)
    	params_vec (numpy.ndarray) : shape = (4, ), [theta_1, theta_2, p_1, p_2]

	Returns :
    	float. log likelihood
    """
    theta_vec, p_vec = params_to_vec(params_vec)
    R = gp_tools.kernel_mat(xmat, theta_vec, p_vec)
    n = R.shape[0]
    Rinv = np.linalg.inv(R)
    detR = np.linalg.det(R)
    hat_sigz_sqr = hat_sigmaz_sqr_mle(y, R)
    print("sigma " + str(hat_sigz_sqr))
    print("Det " + str(detR))
    return - 0.5 * (n * math.log(hat_sigz_sqr) + math.log(detR))


def log_likelihood_fixedp(xmat, y, theta_vec):
    """
    Log likelihood, params_vec = [theta_1, theta_2, p]

    Args :
        xmat (numpy.ndarray) : shape = (n, 2)
    	y (numpy.ndarray) : shape = (n, 1)
    	params_vec (numpy.ndarray) : shape = (2, ) [theta_1, theta_2]

	Returns :
    	float. log likelihood
    """
    p_vec = [1.0, 1.0]
    R = gp_tools.kernel_mat(xmat, theta_vec, p_vec)
    n = R.shape[0]
    Rinv = np.linalg.inv(R)
    detR = np.linalg.det(R)
    hat_sigz_sqr = hat_sigmaz_sqr_mle(y, R)
    print("Theta vec" + str(theta_vec))
    print("sigma " + str(hat_sigz_sqr))
    print("Det " + str(detR))
    return - 0.5 * (n * math.log(hat_sigz_sqr) + math.log(detR))


def max_log_likelihood(xmat, y, xinit):
    # We have an issue with the determinant which gets very
    # very small to the point that a math domain error is raised
    def minus_llk_opti(params): return - log_likelihood_fixedp(xmat, y, params)
    #opt = optimize.minimize(fun=minus_llk_opti, x0=xinit, method="L-BFGS-B")
    opti = optimize.minimize(fun=minus_llk_opti, x0=xinit, method="L-BFGS-B")
    return opti
