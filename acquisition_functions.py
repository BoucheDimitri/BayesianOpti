import numpy as np
import math
import scipy.stats as stats
import prediction_formulae as pred 

def fmin(y):
    return min(y)

def EI(y, xnew, rx, Rinv, beta_hat, function2Bmin):
    f_min = fmin(y)
    y_hat = function2Bmin(xnew)
    sigma_hat = math.sqrt(pred.sigma_est(y, rx, Rinv, beta_hat))
    print(sigma_hat)
    z = (f_min-y_hat)/sigma_hat
    if sigma_hat == 0:
        EI = 0
    else:
        EI = (f_min-y_hat)*stats.norm.cdf(z)+sigma_hat*stats.norm.pdf(z)
    return EI