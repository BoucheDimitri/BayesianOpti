import numpy as np
import scipy.optimize as optimize

import acquisition_functions as af
import exp_kernel
import prediction_formulae as pred


def acq_func(
        xmat,
        xnew,
        y,
        Rinv,
        beta_hat,
        theta,
        p,
        xi=0,
        func_key="EI",
        **kwargs):
    """
    Generate acquisition function for optimization with possibility
    to change easily of acquisition function

    Args:
        xmat (numpy.ndarray) : the data points, shape = (n, k)
        xnew (numpy.ndarray) : the new data point, shape = (k, )
        y (numpy.ndarray) : y, shape=(n, 1)
        Rinv (numpy.ndarray) : Inverse of R, shape=(n, n)
        beta_hat(float) : estimation of beta on the data of xmat
        theta (numpy.ndarray) : vector of theta params, one by dim, shape = (k, )
        p (numpy.ndarray) : powers used to compute the distance, one by dim, shape = (k, )
        xi (float) : Tradeoff parameter between exploration and exploitation
        func_key (str) : Key for acq func, supported : "EI", "GEI", "LCB"
        kwargs : additionnal parameters for acquisition functions (g for GEI for instance)

    Returns:
        float. Value of acquisition function
    """
    rx = exp_kernel.kernel_rx(xmat, xnew, theta, p)
    hat_y = pred.y_est(rx, y, Rinv, beta_hat)
    hat_sigma = np.power(pred.sigma_sqr_est(y, rx, Rinv, beta_hat), 0.5)
    # On pourrait trouver mieux mais vu qu'on va avoir au maximum 5
    # fonction d'acquisition differentes, la liste de conditions reste
    # une solution simple et acceptable
    print("xnew")
    print(xnew)
    if func_key == "EI":
        fmin = np.min(y)
        return af.acq_funcs_dic[func_key](hat_y, hat_sigma, xi, fmin)
    elif func_key == "GEI":
        fmin = np.min(y)
        if "g" not in kwargs.keys():
            g = 1
        else:
            g = kwargs["g"]
        return af.acq_funcs_dic[func_key](hat_y, hat_sigma, xi, fmin, g)
    elif func_key == "LCB":
        return af.acq_funcs_dic[func_key](hat_y, hat_sigma, xi)


def max_acq_func(
        xmat,
        y,
        Rinv,
        beta_hat,
        theta,
        p,
        xinit,
        bounds=None,
        xi=0,
        func_key="EI",
        **kwargs):
    def minus_acq_fun(xnew):
        return float(-acq_func(xmat, xnew, y, Rinv,
                               beta_hat, theta, p, xi, func_key, **kwargs))
    opti = optimize.minimize(
        fun=minus_acq_fun,
        x0=xinit,
        bounds=bounds,
        method='SLSQP')
    return opti
