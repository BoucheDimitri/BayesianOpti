import numpy as np
import math
import scipy.stats as stats
import scipy.optimize as optimize
import prediction_formulae as pred
import gp_tools

def fmin(y):
    return min(y)

def EI(xnew, xtest, y, Rinv, beta_hat, theta_vec, p_vec, function2Bmin):
    f_min = fmin(y)
    y_hat = function2Bmin(xnew)
    rx = gp_tools.kernel_rx_2d_prod(xtest, xnew, theta_vec, p_vec)
    sigma_hat = math.sqrt(pred.sigma_est(y, rx, Rinv, beta_hat))
    if sigma_hat == 0:
        EI = 0
    else:
        z = (f_min-y_hat)/sigma_hat
        EI = float((f_min-y_hat)*stats.norm.cdf(z)+sigma_hat*stats.norm.pdf(z))
    print(EI)
    print(type(EI))
    return EI

def max_EI(xtest, y, Rinv, beta_hat, theta_vec, p_vec, xinit, function2Bmin):
    #bug when nb.it too large => ValueError: Objective function must return a scalar
    def minus_EI(xnew): return float(-EI(xnew, xtest, y, Rinv, beta_hat, theta_vec, p_vec, function2Bmin))
    opti = optimize.minimize(fun=minus_EI, x0=xinit, bounds=((0,0),(5,5)), method='SLSQP')
    return opti