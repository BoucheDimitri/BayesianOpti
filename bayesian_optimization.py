import numpy as np
import cho_inv
import prediction_formulae as pred 
import gp_tools
import test_functions as test_func 
import max_likelihood as max_llk
import acquisition_functions as af

def bayesian_optimization(n, nb_it, p_vec, theta_vec, function2Bmin):
    """
    Function for bayesian optimization with fixed p and theta

    Args:
        n (integer) : number of initial sampling observations
        nb_it (integer) : number of iteration of sampling
        theta_vec (numpy.ndarray) : vector of theta params, one by dim, shape = (2, )
        p_vec (numpy.ndarray) : powers used to compute the distance, one by dim, shape = (2, )

    Returns:
        float. Minimum evaluation of the fonction to be minimized
        numpy.ndarray Point minimizing the function to be minimized
        
	"""
    xtest = 5*np.random.rand(n, 2)
    y = np.zeros((n, 1))
    for i in range(0, n):
    	y[i, 0] = test_func.mystery_vec(xtest[i, :])
    
    for it in range(0,nb_it):

        R = gp_tools.kernel_mat_2d_prod(xtest, theta_vec, p_vec)
        Rinv = cho_inv.cholesky_inv(R)
        beta = pred.beta_est(y, Rinv)
        xinit = xtest[np.argmin(y),]
        optiEI = af.max_EI(xtest, y, Rinv, beta, theta_vec, p_vec, xinit, function2Bmin)
        xnew = optiEI["x"].reshape(1,2)
        ynew = optiEI["fun"].reshape(1,1)
        xtest = np.concatenate((xtest, xnew), axis=0)
        y = np.concatenate((y, ynew))
        
    return min(y), xtest[np.argmin(y),]
    
