import numpy as np
import cho_inv
import prediction_formulae as pred
import exp_kernel
import test_functions as test_func
import max_likelihood as max_llk
import acquisition_functions as af
import acquisition_max as am


def bayesian_optimization(n, nb_it, p_vec, theta_vec, function2Bmin):

    # Ce serait bien de faire une fonction pour une iteration puis
    # de boucler en utilisant cette fonction "atomique" cf infra
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
    xtest = 5 * np.random.rand(n, 2)
    y = np.zeros((n, 1))
    for i in range(0, n):
        y[i, 0] = test_func.mystery_vec(xtest[i, :])

    for it in range(0, nb_it):

        R = exp_kernel.kernel_mat(xtest, theta_vec, p_vec)
        Rinv = cho_inv.cholesky_inv(R)
        beta = pred.beta_est(y, Rinv)
        xinit = 5 * np.random.rand(1, 2)
        optiEI = af.max_EI(
            xtest,
            y,
            Rinv,
            beta,
            theta_vec,
            p_vec,
            xinit,
            function2Bmin)
        xnew = optiEI["x"].reshape(1, 2)
        ynew = np.array(function2Bmin(xnew.reshape(2, 1))).reshape(1, 1)
        xtest = np.concatenate((xtest, xnew), axis=0)
        y = np.concatenate((y, ynew))
        print(it)

    return min(y), y, xtest[np.argmin(y), ], xtest


def bayesian_search(xmat,
                    y,
                    theta,
                    p,
                    xinit,
                    bounds=None,
                    xi=0,
                    acq_func_key="EI",
                    **kwargs):
    """
    Search best point to sample by maximizing acquisition function

    Args:
        xmat (numpy.ndarray) : the data points so far, shape = (n, k)
        y (numpy.ndarray) : y, shape=(n, 1)
        theta (numpy.ndarray) : vector of theta params, one by dim, shape = (k, )
        p (numpy.ndarray) : powers used to compute the distance, one by dim, shape = (k, )
        xinit (numpy.ndarray) : initial value for acquisition maximization, shape = (k, )
        bounds (tuple) : bounds for acquisition maximization in scipy
        xi (float) : Tradeoff parameter between exploration and exploitation
        acq_func_key (str) : Key for acq func, supported : "EI", "GEI", "LCB"
        kwargs : additionnal parameters for acquisition functions (g for GEI for instance)

    Returns:
        numpy.ndarray. The new point to sample from given the data so far
    """
    R = exp_kernel.kernel_mat(xmat, theta, p)
    Rinv = cho_inv.cholesky_inv(R)
    beta_hat = pred.beta_est(y, Rinv)
    opti_result = am.max_acq_func(xmat,
                                  y,
                                  Rinv,
                                  beta_hat,
                                  theta,
                                  p,
                                  xinit,
                                  bounds,
                                  xi,
                                  acq_func_key=acq_func_key,
                                  **kwargs)
    best_xnew = opti_result.x
    print(best_xnew.shape)
    return best_xnew


def evaluate_add(xmat, xnew, y, test_func_key="Mystery"):
    """
    Once point to sample from is chosen, this function evaluate
    the function at that point and update xmat and y to incorporate it

    Args:
        xmat (numpy.ndarray) : the data points so far, shape = (n, k)
        y (numpy.ndarray) : y, shape=(n, 1)
        test_func_key (str) : key of function that we are trying to minimize

    Returns:
         tuple. xmat expanded, y expanded
    """
    ynew = np.array([test_func.funcs_dic[test_func_key](xnew)])
    ynew = ynew.reshape((1, 1))
    y = np.concatenate((y, ynew))
    k = xmat.shape[1]
    xnew = xnew.reshape(1, k)
    xmat = np.concatenate((xmat, xnew))
    return xmat, y


def x_init_indounds(bounds):
    k = len(bounds)
    xinit = np.zeros((k, ))
    for b in range(0, k):
        xinit[b] = np.random.uniform(bounds[b][0], bounds[b][1])
    return xinit


def bayesian_opti(xmat,
                  y,
                  n_it,
                  theta,
                  p,
                  bounds=None,
                  xi=0,
                  acq_func_key="EI",
                  test_func_key="Mystery",
                  **kwargs):
    k = xmat.shape[1]
    for i in range(0, n_it):
        if bounds:
            xinit = x_init_indounds(bounds)
        else:
            xinit = np.random.rand(k)
        xnew = bayesian_search(xmat,
                               y,
                               theta,
                               p,
                               xinit,
                               bounds=bounds,
                               xi=xi,
                               acq_func_key=acq_func_key,
                               **kwargs)
        xmat, y = evaluate_add(xmat, xnew, y, test_func_key=test_func_key)
    return xmat, y
