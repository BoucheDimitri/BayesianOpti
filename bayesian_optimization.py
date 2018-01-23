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
                    func_key="EI",
                    **kwargs):
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
                                  func_key=func_key,
                                  **kwargs)
    best_xnew = opti_result.x
    return best_xnew
