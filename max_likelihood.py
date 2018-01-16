import numpy as np
import prediction_formulae as pred 
import gp_tools
import cho_inv


def hat_sigmaz_sqr_mle(xmat, y, params_vec):
	"""
	Since R depends on theta and p, so does Rinv and beta estimate
	We need to have hat_sigmaz as a function of theta and p for MLE

	Args :
        xmat (numpy.ndarray) : shape = (n, 2)
        y (numpy.ndarray) : shapre = (n, 1)
        params_vec (numpy.ndarray) : shape = (3, ), [theta_1, theta_2, p]
	"""
	theta_vec = params_vec[0:2]
	p = float(params_vec[2])
	R = gp_tools.kernel_mat(xmat, theta_vec, p)
	Rinv = cho_inv.cholesky_inv(R)
	hat_beta = pred.beta_est_bis(y, Rinv)
	return pred.hat_sigmaz_sqr(y, Rinv, hat_beta)
